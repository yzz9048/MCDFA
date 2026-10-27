import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from Models.lstm_encdec.model import EncDecAD
from Utils.Data_utils.real_datasets import CustomDataset
from Utils.reconstruction_report import (
    compute_point_metrics,
    compute_window_metrics,
    fit_gaussian,
    mahalanobis_scores,
    select_threshold,
    window_scores_from_points,
    write_reconstruction_report,
)
from engine.encdec_solver import EncDecTrainer
from main import parse_args, resolve_encdec_config


def make_toy_csv(path, length=400, features=3, fault_ranges=()):
    timeline = np.arange(length, dtype=np.int64)
    data = {'timestamp': timeline}
    for index in range(features):
        data[f'sensor_{index}'] = (
            np.sin(timeline / (7.0 + index)) + 0.01 * timeline + index
        )
    labels = np.zeros(length, dtype=np.int64)
    for start, end in fault_ranges:
        labels[start:end] = 1
    data['label'] = labels
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def make_reconstruction_dataset(root, csv, period, normal_only=False, save=False):
    return CustomDataset(
        name='toy',
        data_root=str(csv),
        window=20,
        DeltaT=1,
        proportion=0.6,
        validation_proportion=0.2,
        period=period,
        task='reconstruction',
        normal_only=normal_only,
        feature_scaling='train_minmax',
        save2npy=save,
        neg_one_to_one=True,
        output_dir=str(root),
    )


class ReconstructionDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = make_toy_csv(
            self.root / 'toy.csv',
            length=240,
            fault_ranges=((90, 96), (150, 156), (205, 211)),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_windows_respect_chronological_split_boundaries(self):
        train = make_reconstruction_dataset(self.root, self.csv, 'train', normal_only=True)
        validation = make_reconstruction_dataset(self.root, self.csv, 'validation')
        test = make_reconstruction_dataset(self.root, self.csv, 'test')

        all_starts = set(train.all_sample_starts.tolist())
        for dataset in (train, validation, test):
            self.assertTrue(set(dataset.sample_starts.tolist()).issubset(all_starts))
        starts = [set(dataset.sample_starts.tolist()) for dataset in (train, validation, test)]
        self.assertFalse(starts[0] & starts[1])
        self.assertFalse(starts[0] & starts[2])
        self.assertFalse(starts[1] & starts[2])

        self.assertTrue((train.sample_starts + train.window <= train.split_timestamp).all())
        self.assertTrue((validation.sample_starts >= validation.split_timestamp).all())
        self.assertTrue(
            (validation.sample_starts + validation.window <= validation.validation_timestamp).all()
        )
        self.assertTrue((test.sample_starts >= test.validation_timestamp).all())
        self.assertEqual(
            train.boundary_crossing_windows_excluded,
            len(train.all_sample_starts) - sum(train.split_window_counts.values()),
        )
        in_train = train.all_sample_starts + train.window <= train.split_timestamp
        self.assertEqual(
            train.split_window_counts['train'] - train.sample_num,
            int(np.count_nonzero(train.all_window_labels[in_train])),
        )

    def test_training_split_contains_only_fault_free_windows(self):
        train = make_reconstruction_dataset(self.root, self.csv, 'train', normal_only=True)
        self.assertTrue(np.all(train.window_labels == 0))
        self.assertFalse(np.any(train.labels != 0))
        self.assertEqual(train.samples.shape[1:], (train.window, 3))
        self.assertTrue(np.isfinite(train.samples).all())
        self.assertGreaterEqual(train.samples.min(), -1.0)
        self.assertLessEqual(train.samples.max(), 1.0)
        self.assertEqual(train.__getitem__(0)[0].shape, (train.window, 3))

    def test_normal_only_is_rejected_outside_training(self):
        with self.assertRaises(ValueError):
            make_reconstruction_dataset(self.root, self.csv, 'validation', normal_only=True)

    def test_normalization_fits_only_the_training_segment_and_is_shared(self):
        train_a = make_reconstruction_dataset(self.root, self.csv, 'train', save=True)
        train_b = make_reconstruction_dataset(self.root, self.csv, 'train', save=True)
        test = make_reconstruction_dataset(self.root, self.csv, 'test', save=True)

        raw = pd.read_csv(self.csv).to_numpy(dtype=np.float64)
        training_features = raw[:train_a.split_timestamp, 1:-1]
        expected_min = training_features.min(axis=0)
        expected_max = training_features.max(axis=0)
        np.testing.assert_allclose(train_a.scaler.data_min_, expected_min)
        np.testing.assert_allclose(train_a.scaler.data_max_, expected_max)
        np.testing.assert_allclose(test.scaler.data_min_, expected_min)
        np.testing.assert_allclose(test.scaler.data_max_, expected_max)
        self.assertAlmostEqual(train_a.samples.min(), -1.0)
        self.assertAlmostEqual(train_a.samples.max(), 1.0)

        state_a = json.loads(
            (self.root / 'samples' / 'toy_normalization.json').read_text(encoding='utf-8')
        )
        self.assertEqual(state_a['feature_scaling'], 'train_minmax')
        self.assertTrue(state_a['neg_one_to_one'])
        self.assertEqual(train_a.normalization_state(), train_b.normalization_state())
        self.assertEqual(train_a.normalization_state(), test.normalization_state())
        self.assertTrue(
            (self.root / 'samples' / 'toy_reconstruction_truth_20_test.npy').is_file()
        )
        self.assertTrue(
            (self.root / 'samples' / 'toy_norm_truth_20_validation.npy').is_file()
        )

class EncDecModelTests(unittest.TestCase):
    def test_forward_shapes_and_finite_loss(self):
        model = EncDecAD(feature_size=4, hidden_size=8, num_layers=2, dropout=0.1)
        x = torch.randn(5, 7, 4)
        reconstruction = model(x)
        self.assertEqual(reconstruction.shape, x.shape)
        self.assertTrue(torch.isfinite(reconstruction).all())
        loss = torch.nn.functional.mse_loss(reconstruction, x)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))

        inference = model.reconstruct(x)
        self.assertEqual(inference.shape, x.shape)
        self.assertTrue(torch.isfinite(inference).all())

    def test_single_step_window_and_invalid_inputs(self):
        model = EncDecAD(feature_size=2, hidden_size=4)
        self.assertEqual(model(torch.randn(3, 1, 2)).shape, (3, 1, 2))
        with self.assertRaises(ValueError):
            model(torch.randn(3, 4, 3))
        with self.assertRaises(ValueError):
            EncDecAD(feature_size=0)
        with self.assertRaises(ValueError):
            EncDecAD(feature_size=2, dropout=1.0)


class ReconstructionReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_gaussian_and_mahalanobis_scores(self):
        normal_errors = np.zeros((10, 2), dtype=np.float64)
        stats = fit_gaussian(normal_errors, regularization=1.0e-3)
        np.testing.assert_allclose(stats['mean'], 0.0)
        np.testing.assert_allclose(stats['covariance'], 1.0e-3 * np.eye(2))
        self.assertEqual(stats['sample_count'], 10)

        errors = np.array([[[0.0, 0.0], [1.0, 0.0]], [[5.0, 5.0], [0.0, 3.0]]])
        scores = mahalanobis_scores(errors, stats)
        self.assertEqual(scores.shape, (2, 2))
        self.assertAlmostEqual(scores[0, 0], 0.0)
        self.assertGreater(scores[0, 1], 0.0)
        self.assertGreater(scores[1, 0], scores[0, 1])

    def test_threshold_selection_maximises_fbeta(self):
        scores = np.array([[0.1, 0.2], [0.3, 5.0], [6.0, 0.2]])
        labels = np.array([[0, 0], [0, 1], [1, 0]])
        selection = select_threshold(scores, labels, beta=0.05)
        self.assertGreaterEqual(selection['selected_threshold'], 0.3)
        self.assertLessEqual(selection['selected_threshold'], 5.0)
        self.assertEqual(selection['selected_metrics']['true_positive'], 2)
        self.assertEqual(selection['selected_metrics']['false_positive'], 0)
        self.assertEqual(selection['selected_metrics']['fbeta'], 1.0)

        point_metrics = compute_point_metrics(scores, labels, selection['selected_threshold'])
        self.assertEqual(point_metrics['count'], 6)
        self.assertEqual(point_metrics['true_positive'], 2)

    def test_threshold_fallback_without_validation_positives(self):
        scores = np.linspace(0.0, 1.0, 101)
        labels = np.zeros_like(scores)
        selection = select_threshold(
            scores, labels, beta=0.05, fallback_false_positive_rate=0.1
        )
        self.assertIn('fallback', selection['selection_rule'])
        self.assertTrue(np.isfinite(selection['selected_threshold']))
        self.assertEqual(selection['selected_metrics']['actual_positive'], 0)

    def test_window_metrics_preserve_the_point_threshold(self):
        scores = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 0.1]])
        labels = np.array([[0, 0], [1, 0], [0, 0]])
        metrics = compute_window_metrics(scores, labels, threshold=1.5, beta=0.05)
        self.assertEqual(metrics['threshold'], 1.5)
        self.assertEqual(metrics['true_positive'], 1)
        self.assertEqual(metrics['false_positive'], 0)
        self.assertEqual(metrics['precision'], 1.0)
        self.assertEqual(metrics['recall'], 1.0)

    def test_window_score_reductions(self):
        scores = np.arange(20, dtype=np.float64).reshape(2, 10)
        np.testing.assert_allclose(window_scores_from_points(scores, 'max'), [9.0, 19.0])
        np.testing.assert_allclose(window_scores_from_points(scores, 'mean'), [4.5, 14.5])
        np.testing.assert_allclose(
            window_scores_from_points(scores, 'topk_mean'), [9.0, 19.0]
        )
        with self.assertRaises(ValueError):
            window_scores_from_points(scores, 'sum')

    def test_report_writes_scores_metrics_and_threshold_selection(self):
        csv = make_toy_csv(
            self.root / 'toy.csv', length=240, fault_ranges=((150, 156), (205, 211))
        )
        dataset = make_reconstruction_dataset(self.root, csv, 'validation')
        rng = np.random.default_rng(0)
        scores = rng.uniform(0.0, 1.0, size=dataset.labels.shape)
        scores[dataset.labels != 0] += 5.0
        selection = select_threshold(scores, dataset.labels, beta=0.05)
        stats = fit_gaussian(np.zeros((4, 3)), regularization=1.0e-3)
        summary = write_reconstruction_report(
            dataset=dataset,
            point_scores=scores,
            threshold=selection['selected_threshold'],
            beta=0.05,
            output_dir=str(self.root / 'reports'),
            split='validation',
            config={'save_point_scores': True, 'window_score_reduction': 'max'},
            stats=stats,
            checkpoint='checkpoint-best.pt',
            checkpoint_step=7,
            threshold_selection=selection,
        )

        reports = self.root / 'reports'
        self.assertTrue((reports / 'toy_window_scores_validation.csv').is_file())
        self.assertTrue((reports / 'toy_point_scores_validation.npz').is_file())
        self.assertTrue((reports / 'toy_threshold_sweep_validation.csv').is_file())
        self.assertTrue((reports / 'toy_threshold_selection.json').is_file())
        self.assertTrue((reports / 'toy_detection_summary_validation.json').is_file())
        self.assertTrue((reports / 'toy_selected_threshold_metrics.csv').is_file())

        payload = json.loads(
            (reports / 'toy_detection_summary_validation.json').read_text(encoding='utf-8')
        )
        self.assertEqual(payload['task'], 'reconstruction')
        self.assertEqual(payload['threshold'], selection['selected_threshold'])
        self.assertEqual(payload['metrics']['point']['count'], dataset.labels.size)
        self.assertEqual(payload['metrics']['window']['count'], dataset.sample_num)
        self.assertEqual(
            payload['metrics']['point']['true_positive'],
            summary['metrics']['point']['true_positive'],
        )
        window_frame = pd.read_csv(reports / 'toy_window_scores_validation.csv')
        self.assertEqual(len(window_frame), dataset.sample_num)
        self.assertIn('decision_threshold', window_frame.columns)


class EncDecTrainerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = make_toy_csv(
            self.root / 'toy.csv',
            length=400,
            features=3,
            fault_ranges=((100, 104), (260, 266), (340, 346)),
        )

    def tearDown(self):
        self.temp.cleanup()

    def config(self, results_folder):
        return {
            'model': {
                'target': 'Models.lstm_encdec.model.EncDecAD',
                'params': {
                    'feature_size': 3,
                    'hidden_size': 8,
                    'num_layers': 1,
                    'dropout': 0.0,
                },
            },
            'solver': {
                'base_lr': 1.0e-2,
                'max_epochs': 2,
                'gradient_accumulate_every': 1,
                'early_stopping_patience': 2,
                'results_folder': str(results_folder),
            },
            'dataloader': {
                'target': 'Utils.Data_utils.real_datasets.CustomDataset',
                'params': {
                    'name': 'toy',
                    'data_root': str(self.csv),
                    'window': 20,
                    'DeltaT': 1,
                    'proportion': 0.6,
                    'validation_proportion': 0.2,
                    'task': 'reconstruction',
                    'normal_only': True,
                    'feature_scaling': 'train_minmax',
                    'save2npy': False,
                    'neg_one_to_one': True,
                    'seed': 123,
                    'period': 'train',
                },
                'batch_size': 16,
                'beta': 0.05,
                'covariance_regularization': 1.0e-3,
                'fallback_false_positive_rate': 0.01,
                'window_score_reduction': 'max',
                'validation_fit_ratio': 0.5,
                'save_point_scores': False,
            },
        }

    def dataset(self, period, normal_only=False):
        return make_reconstruction_dataset(
            self.root, self.csv, period, normal_only=normal_only
        )

    def test_train_save_and_evaluate_validation_and_test(self):
        results = self.root / 'checkpoints'
        reports = self.root / 'reports'
        config = self.config(results)
        args = SimpleNamespace(save_dir=str(reports), checkpoint='best')

        train_info = {'dataset': self.dataset('train', normal_only=True), 'dataloader': None}
        train_info['dataloader'] = torch.utils.data.DataLoader(
            train_info['dataset'], batch_size=16, shuffle=False
        )
        validation_dataset = self.dataset('validation')
        validation_info = {
            'dataset': validation_dataset,
            'dataloader': torch.utils.data.DataLoader(
                validation_dataset, batch_size=16, shuffle=False
            ),
        }
        model = EncDecAD(feature_size=3, hidden_size=8)
        trainer = EncDecTrainer(
            config, args, model, train_info, validation_info=validation_info
        )
        trainer.train()

        self.assertTrue((results / 'checkpoint-best.pt').is_file())
        self.assertEqual(trainer.epoch, 2)
        self.assertGreater(len(trainer.validation_fit_indices), 0)
        self.assertFalse(
            set(trainer.validation_fit_indices.tolist())
            & set(trainer.validation_selection_indices.tolist())
        )
        self.assertTrue(
            np.all(validation_dataset.window_labels[trainer.validation_fit_indices] == 0)
        )

        test_dataset = self.dataset('test')
        test_info = {
            'dataset': test_dataset,
            'dataloader': torch.utils.data.DataLoader(
                test_dataset, batch_size=16, shuffle=False
            ),
        }
        reloaded_model = EncDecAD(feature_size=3, hidden_size=8)
        evaluator = EncDecTrainer(
            config,
            args,
            reloaded_model,
            validation_info={
                'dataset': validation_dataset,
                'dataloader': validation_info['dataloader'],
            },
        )
        summary = evaluator.evaluate(
            validation_info={
                'dataset': validation_dataset,
                'dataloader': validation_info['dataloader'],
            },
            test_info=test_info,
        )

        self.assertTrue(np.isfinite(evaluator.decision_threshold))
        self.assertGreaterEqual(evaluator.decision_threshold, 0.0)
        self.assertEqual(summary['split'], 'test')
        self.assertTrue((reports / 'toy_window_scores_validation.csv').is_file())
        self.assertTrue((reports / 'toy_window_scores_test.csv').is_file())
        self.assertTrue((reports / 'toy_detection_summary_validation.json').is_file())
        self.assertTrue((reports / 'toy_detection_summary_test.json').is_file())
        self.assertTrue((reports / 'toy_selected_threshold_metrics.csv').is_file())
        self.assertEqual(summary['threshold'], evaluator.decision_threshold)
        self.assertEqual(
            summary['metrics']['point']['count'], test_dataset.labels.size
        )
        selected = pd.read_csv(reports / 'toy_selected_threshold_metrics.csv')
        self.assertEqual(set(selected['split']), {'validation', 'test'})
        np.testing.assert_allclose(selected['threshold'], evaluator.decision_threshold)

    def test_training_requires_a_validation_split(self):
        config = self.config(self.root / 'checkpoints')
        args = SimpleNamespace(save_dir=str(self.root / 'reports'), checkpoint='best')
        train_dataset = self.dataset('train', normal_only=True)
        train_info = {
            'dataset': train_dataset,
            'dataloader': torch.utils.data.DataLoader(
                train_dataset, batch_size=16, shuffle=False
            ),
        }
        trainer = EncDecTrainer(
            config, args, EncDecAD(feature_size=3, hidden_size=8), train_info
        )
        with self.assertRaises(ValueError):
            trainer.train()


class EncDecConfigTests(unittest.TestCase):
    def test_resolves_dedicated_configs(self):
        self.assertTrue(resolve_encdec_config('GAIA').endswith('encdec-gaia.yaml'))
        self.assertTrue(resolve_encdec_config('sn').endswith('encdec-sn.yaml'))
        self.assertTrue(resolve_encdec_config('media').endswith('encdec-media.yaml'))
        self.assertTrue(resolve_encdec_config('maat').endswith('encdec-maat.yaml'))
        with self.assertRaises(FileNotFoundError):
            resolve_encdec_config('does-not-exist')

    def test_cli_resolves_the_encdec_config(self):
        args = parse_args(['--name', 'sn'])
        self.assertTrue(args.config_file.endswith('encdec-sn.yaml'))
        self.assertEqual(args.checkpoint, 'best')
        self.assertEqual(args.split, 'test')
        self.assertTrue(args.save_dir.endswith('sn'))

    def test_cli_rejects_unknown_dataset_without_a_config(self):
        with self.assertRaises(FileNotFoundError):
            parse_args(['--name', 'does-not-exist'])


if __name__ == '__main__':
    unittest.main()
