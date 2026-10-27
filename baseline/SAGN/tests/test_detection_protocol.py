import contextlib
import io
import json
import random
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from Data.build_dataloader import build_dataloader_cond
from Utils.Data_utils.real_datasets import CustomDataset
from Utils.detection_report import event_metrics, select_validation_threshold, write_detection_report
from Utils.lead_time import compute_event_classification_metrics
from Utils.io_utils import load_yaml_config
from engine.solver import Trainer
from main import parse_args


class TinyForecast(nn.Module):
    """Cheap sampler exercising the real trainer/report integration."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.2))
        self.register_buffer('betas', torch.tensor([0.1]))
        self.seq_length = 30
        self.num_timesteps = 1
        self.received_labels = []

    def forward(self, data, labels, DeltaT, clip, granger):
        self.received_labels.append(labels.detach().clone())
        return self.weight.square(), granger.weight.square().mean()

    def sample_infill(self, granger, shape, history, **kwargs):
        causal_loss = granger.weight.square().mean()
        causal_loss.backward()
        score = history[:, -1, 1] * 0.01 + granger.weight.sum() * 0.2 + torch.rand(len(history))
        logits = torch.stack([torch.zeros_like(score), score], dim=1)
        return torch.zeros(shape), logits, causal_loss.item()


class DetectionProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / 'data.csv'
        labels = np.zeros(201)
        labels[30:35] = 1
        labels[127:132] = 1
        labels[174:178] = 1
        labels[184:188] = 1
        pd.DataFrame({
            'timestamp': np.arange(201), 'rps': np.arange(201) + 1,
            'value': np.arange(201) + 2, 'label': labels,
        }).to_csv(self.csv, index=False)
        self.args = SimpleNamespace(pred_len=10, clip=True, save_dir=str(self.root / 'reports'))

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, period, save=False):
        return CustomDataset(
            name='toy', data_root=str(self.csv), window=30, DeltaT=1,
            proportion=0.6, validation_proportion=0.2, forecast_length=10,
            predict_length=None if period == 'train' else 10, period=period,
            save2npy=save, neg_one_to_one=False, output_dir=str(self.root),
        )

    def config(self, period='test'):
        return {
            'model': {'params': {'DeltaT': 10}},
            'dataloader': {
                'target': 'Utils.Data_utils.real_datasets.CustomDataset',
                'params': {'name': 'toy', 'data_root': str(self.csv), 'window': 30,
                           'DeltaT': 1, 'proportion': 0.6, 'validation_proportion': 0.2,
                           'period': period, 'save2npy': False, 'neg_one_to_one': False},
                'batch_size': 8, 'decision_threshold': 0.5,
                'select_threshold_on_validation': True,
                'threshold_selection_objective': 'balanced',
            },
            'solver': {
                'max_epochs': 1, 'gradient_accumulate_every': 1, 'save_cycle': 1,
                'results_folder': str(self.root / 'checkpoints'), 'base_lr': 0.01,
                'ema': {'decay': 0.9, 'update_interval': 1},
                'scheduler': {'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                              'params': {'warmup': 5, 'warmup_lr': 0.05, 'min_lr': 0.0}},
            },
        }

    def trainer(self, dataset, loader=None):
        loader = loader or DataLoader(dataset, batch_size=8, shuffle=False)
        return Trainer(self.config(dataset.period), self.args, TinyForecast(), nn.Linear(1, 1),
                       {'dataset': dataset, 'dataloader': loader})

    def test_raw_timeline_splits_have_disjoint_targets_and_complete_windows(self):
        splits = [self.dataset(period, save=True) for period in ('train', 'validation', 'test')]
        target_sets = []
        for dataset in splits:
            target_sets.append({int(start + offset) for start in dataset.sample_starts for offset in range(20, 30)})
            self.assertTrue(np.all(np.diff(dataset.sample_starts) == 1))
            for row, start in enumerate(dataset.sample_starts):
                np.testing.assert_array_equal(dataset.samples[row], dataset.rawdata[start:start + 30, :-1])
            self.assertEqual(dataset.split_window_counts, {'train': 92, 'validation': 31, 'test': 31})
            self.assertEqual(dataset.boundary_crossing_windows_excluded, 18)
        self.assertFalse(target_sets[0] & target_sets[1])
        self.assertFalse(target_sets[1] & target_sets[2])
        self.assertEqual(max(target_sets[0]), 120)
        self.assertEqual((min(target_sets[1]), max(target_sets[1])), (121, 160))
        self.assertEqual((min(target_sets[2]), max(target_sets[2])), (161, 200))
        self.assertTrue((splits[1].sample_starts < splits[1].split_timestamp).any())
        for dataset in splits[1:]:
            self.assertTrue(dataset.masking[:, :-10].all())
            self.assertFalse(dataset.masking[:, -10:].any())
        for period in ('train', 'validation', 'test'):
            self.assertTrue((self.root / 'samples' / f'toy_norm_truth_30_{period}.npy').is_file())

    def test_loader_keeps_inference_chronological_and_training_mask_free(self):
        config = self.config('train')
        before = deepcopy(config)
        info = build_dataloader_cond(config, self.args)
        self.assertIsNone(info['dataset'].pred_len)
        self.assertEqual(info['dataset'].forecast_length, 10)
        self.assertEqual(config, before)
        config['dataloader']['params']['period'] = 'validation'
        config['dataloader']['shuffle'] = True
        info = build_dataloader_cond(config, self.args)
        self.assertIsInstance(info['dataloader'].sampler, torch.utils.data.SequentialSampler)

    def test_training_ignores_faults_that_exist_only_in_history(self):
        dataset = self.dataset('train')
        index = 17  # history includes 30..34; forecast is 37..46, all normal.
        self.assertTrue((dataset.labels[index] != 0).any())
        loader = DataLoader(TensorDataset(
            torch.tensor(dataset.samples[index:index + 1], dtype=torch.float32),
            torch.tensor(dataset.labels[index:index + 1]),
        ))
        trainer = self.trainer(dataset, loader)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            trainer.train()
        self.assertEqual(trainer.model.received_labels[0].shape, (1, 10))
        self.assertFalse(trainer.model.received_labels[0].bool().any())
        trainer.load('best')

    def test_persistent_alarm_cannot_match_multiple_events(self):
        labels = np.zeros(35)
        labels[10:13] = labels[20:23] = labels[25:28] = 1
        starts = np.arange(24)
        metrics = compute_event_classification_metrics(np.ones(24), starts, labels, 8, 3)
        self.assertEqual(metrics['eligible_onsets'], 3)
        self.assertEqual(metrics['matched_alarm_events'], 1)
        self.assertEqual(metrics['detected_alarm_events'], 1)
        self.assertAlmostEqual(metrics['event_recall'], 1 / 3)
        self.assertEqual(metrics['event_avg_lead_time_timestamps'], 3)
        self.assertEqual(metrics['event_avg_lead_time_all_eligible_onsets_timestamps'], 1)

    def test_report_preserves_false_alarms_and_matches_reference_metrics(self):
        dataset = self.dataset('test')
        probabilities = np.zeros(len(dataset))
        probabilities[0] = 0.8  # observation 160 predicts 161..170: false alarm.
        probabilities[4] = 0.9  # observation 164 predicts onset 174: 10-step lead.
        probabilities[14] = 0.9  # observation 174 predicts onset 184: 10-step lead.
        labels = (dataset.labels[:, -10:] != 0).any(axis=1).astype(int)
        report = write_detection_report(dataset, probabilities, labels, 10, self.root,
                                        self.config()['dataloader'], threshold=0.5)
        metrics = report['metrics']
        reference = compute_event_classification_metrics(probabilities >= 0.5, dataset.sample_starts,
                                                          dataset.raw_labels, 30, 10)
        for key in ('event_precision', 'event_recall', 'event_f1', 'matched_alarm_events',
                    'false_alarm_events', 'event_avg_lead_time_timestamps'):
            self.assertEqual(metrics[key], reference[key])
        self.assertEqual(metrics['false_alarm_events'], 1)
        self.assertAlmostEqual(metrics['event_precision'], 2 / 3)
        self.assertEqual(metrics['event_recall'], 1)
        self.assertEqual(metrics['event_avg_lead_time_timestamps'], 10)
        alarms = pd.read_csv(self.root / 'toy_alarm_event_report_test.csv')
        self.assertEqual(int(alarms['is_false_alarm_event'].sum()), 1)

    def test_validation_selection_rejects_test_rows(self):
        rows = [{'split': 'test'}]
        with self.assertRaisesRegex(ValueError, 'validation-only'):
            select_validation_threshold(rows, np.zeros(50), np.arange(50), {})

    def test_threshold_nextafter_is_not_rounded_to_float32(self):
        dataset = self.dataset('validation')
        probability = float(np.float32(0.25))
        labels = (dataset.labels[:, -10:] != 0).any(axis=1).astype(int)
        config = self.config()['dataloader']
        config['select_threshold_on_validation'] = False
        report = write_detection_report(dataset, np.full(len(dataset), probability), labels, 10,
                                        self.root, config, threshold=np.nextafter(probability, np.inf))
        self.assertEqual(report['metrics']['predicted_positive_windows'], 0)
        self.assertEqual(len(pd.read_csv(self.root / 'toy_alarm_event_report_validation.csv')), 0)
        json.loads((self.root / 'toy_detection_summary_validation.json').read_text())

    def test_validation_does_not_modify_test_optimizer_weights_or_randomness(self):
        validation, test = self.dataset('validation'), self.dataset('test')
        trainer = self.trainer(test)
        initial_weights = deepcopy(trainer.granger.state_dict())
        initial_scheduler = deepcopy(trainer.schG.state_dict())
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        expected_random = (random.random(), np.random.rand(), torch.rand(1))
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        with contextlib.redirect_stdout(io.StringIO()):
            with trainer.validation_prediction_context():
                trainer.restore(DataLoader(validation, batch_size=8), sampling_steps=1)
                self.assertGreater(trainer.schG.last_epoch, initial_scheduler['last_epoch'])
        for key, value in initial_weights.items():
            torch.testing.assert_close(trainer.granger.state_dict()[key], value)
        self.assertEqual(trainer.schG.state_dict(), initial_scheduler)
        self.assertFalse(trainer.optG.state)
        self.assertEqual(random.random(), expected_random[0])
        self.assertEqual(np.random.rand(), expected_random[1])
        torch.testing.assert_close(torch.rand(1), expected_random[2])
        selected = trainer.decision_threshold
        self.assertIsNotNone(trainer.last_detection_report['validation_threshold_selection'])
        with contextlib.redirect_stdout(io.StringIO()):
            samples = trainer.restore(DataLoader(test, batch_size=8), sampling_steps=1)
        self.assertEqual(samples.shape, (len(test), 10, 1))
        self.assertEqual(trainer.last_detection_report['threshold'], selected)
        self.assertIsNone(trainer.last_detection_report['validation_threshold_selection'])
        selected_metrics = pd.read_csv(Path(self.args.save_dir) / 'toy_selected_threshold_metrics.csv')
        self.assertEqual(selected_metrics['split'].tolist(), ['validation', 'test'])
        self.assertEqual(selected_metrics['threshold'].nunique(), 1)

    def test_cli_resolves_each_dataset_and_leaves_gaia_on_its_original_config(self):
        for name in ('sn', 'media', 'maat'):
            args = parse_args(['--name', name, '--pred_len', '10'])
            self.assertEqual(Path(args.config_file).name, f'predict-{name}.yaml')
        args = parse_args(['--name', 'GAIA', '--pred_len', '10'])
        self.assertEqual(Path(args.config_file).name, 'predict.yaml')

    def test_main_config_reader_accepts_utf8_comments_on_windows(self):
        for name in ('sn', 'media', 'maat'):
            args = parse_args(['--name', name, '--pred_len', '10'])
            config = load_yaml_config(args.config_file)
            self.assertEqual(config['dataloader']['params']['DeltaT'], 1)
            self.assertEqual(config['dataloader']['params']['validation_proportion'], 0.2)


if __name__ == '__main__':
    unittest.main()
