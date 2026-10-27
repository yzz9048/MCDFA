import contextlib
import io
import json
import random
import tempfile
import unittest
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from Data.build_dataloader import build_dataloader_cond
from Models.sagn.model import SAGN, DiffusionConv, GumbelCorrelationGraph
from engine.sagn_solver import SAGNTrainer


def tiny_model(**overrides):
    params = dict(
        seq_length=30, DeltaT=10, feature_size=5, hidden_dim=8,
        graph_conv_dim=4, graph_embed_dim=4, graph_steps=1,
        gumbel_temperature=0.5, gumbel_temperature_min=0.1,
        gumbel_anneal_steps=10, eval_temperature=0.1,
    )
    params.update(overrides)
    return SAGN(**params)


class SAGNModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = tiny_model()
        self.history = torch.randn(4, 20, 4)

    def test_sampled_graph_is_deterministic_in_eval_mode(self):
        self.model.eval()
        graph_one = self.model.graph_from_history(self.history)[0]
        graph_two = self.model.graph_from_history(self.history)[0]
        torch.testing.assert_close(graph_one, graph_two)
        self.assertTrue(torch.all(graph_one >= 0) and torch.all(graph_one <= 1))
        self.assertEqual(graph_one.shape, (4, 3, 3))

    def test_gumbel_sampling_varies_during_training(self):
        graph = GumbelCorrelationGraph(feature_count=2, conv_dim=4, embed_dim=4)
        graph.train()
        node_input = torch.randn(4, 20, 3, 2)
        sample_one = graph(node_input, sample=True)[0]
        sample_two = graph(node_input, sample=True)[0]
        self.assertFalse(torch.allclose(sample_one, sample_two))
        adjacent = sample_one.unsqueeze(2) - sample_one.unsqueeze(1)
        # Edge weights come from sigmoid((logit + gum_u - gum_v) / delta).
        self.assertGreaterEqual(float(adjacent.min()), -1.0)
        self.assertLessEqual(float(adjacent.max()), 1.0)

    def test_diffusion_convolution_normalizes_forward_and_backward(self):
        conv = DiffusionConv(input_dim=2, output_dim=3, steps=1)
        graph = torch.tensor([[[0.0, 1.0], [0.5, 0.0]]])
        forward, backward = conv.normalized_adjacency(graph)
        torch.testing.assert_close(forward.sum(dim=-1), torch.ones(1, 2))
        torch.testing.assert_close(backward.sum(dim=-1), torch.ones(1, 2))
        self.assertEqual(conv(torch.randn(1, 2, 2), graph).shape, (1, 2, 3))

    def test_predictions_ignore_future_values_without_teacher_forcing(self):
        self.model.eval()
        prediction = self.model.predict(self.history, steps=3)
        teacher_free, _ = self.model(self.history, target=None, steps=3)
        torch.testing.assert_close(prediction, teacher_free)
        teacher_forced, _ = self.model(
            self.history, torch.randn(4, 10, 3), steps=3)
        # Teacher forcing may only affect steps after the first forecast.
        torch.testing.assert_close(prediction[:, 0], teacher_forced[:, 0])
        self.assertEqual(tuple(prediction.shape), (4, 3, 3))

    def test_residual_score_is_bounded_and_monotone(self):
        errors = torch.tensor([
            [0.1, 0.2, 0.3],
            [0.2, 0.2, 0.3],
            [0.3, 0.2, 0.3],
            [0.15, 0.2, 0.3],
            [0.1, 0.2, 0.3],
        ])
        self.model.calibrate_from_errors(errors)
        center = self.model.score_from_node_errors(torch.tensor([0.15, 0.2, 0.3]))
        low = self.model.score_from_node_errors(torch.tensor([0.2, 0.2, 0.3]))
        high = self.model.score_from_node_errors(torch.tensor([0.9, 0.2, 0.3]))
        self.assertAlmostEqual(float(center), 0.0, places=6)
        self.assertTrue(0.0 < float(low) < float(high) <= 1.0)

    def test_calibration_round_trip(self):
        errors = torch.rand(16, 3) + 0.1
        self.model.calibrate_from_errors(errors)
        saved = self.model.get_calibration()
        restored = tiny_model()
        restored.set_calibration(saved)
        self.assertEqual(restored.get_calibration(), saved)


class SAGNIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv = self.root / 'data.csv'
        labels = np.zeros(201)
        labels[30:35] = 1
        labels[127:132] = 1
        labels[174:178] = 1
        pd.DataFrame({
            'timestamp': np.arange(201),
            'rps': np.arange(201) + 1,
            'a': np.arange(201) + 2,
            'b': np.arange(201) % 7 + 3,
            'c': np.arange(201) % 5 + 4,
            'label': labels,
        }).to_csv(self.csv, index=False)
        self.args = SimpleNamespace(
            pred_len=10, clip=True, save_dir=str(self.root / 'reports'),
        )
        self.config = {
            'model': {'params': {'DeltaT': 10, 'seq_length': 30}},
            'dataloader': {
                'target': 'Utils.Data_utils.real_datasets.CustomDataset',
                'params': {'name': 'toy', 'data_root': str(self.csv),
                           'window': 30, 'DeltaT': 1, 'proportion': 0.6,
                           'validation_proportion': 0.2, 'save2npy': False,
                           'neg_one_to_one': False,
                           'feature_scaling': 'train_minmax'},
                'batch_size': 8, 'prediction_batch_size': 8,
                'decision_threshold': 0.5,
                'select_threshold_on_validation': True,
                'threshold_selection_objective': 'balanced',
            },
            'solver': {
                'max_epochs': 3, 'gradient_accumulate_every': 1,
                'results_folder': str(self.root / 'checkpoints'),
                'base_lr': 0.01,
                'scheduler': {
                    'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                    'params': {'warmup': 1, 'warmup_lr': 0.02, 'min_lr': 0.0},
                },
            },
        }

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self, period):
        config = deepcopy(self.config)
        config['dataloader']['params']['period'] = period
        return build_dataloader_cond(config, self.args)

    def trainer(self):
        model = tiny_model(pool={'size': 2, 'snapshot_interval': 1,
                                 'validation_windows': 8},
                           online={'enabled': True, 'epochs': 1,
                                   'window_batch': 2, 'warmup': 1,
                                   'validation_size': 4, 'pool_update': True},
                           residual_score={'aggregation': 'max', 'steps': 3,
                                           'calibration_windows': 64,
                                           'min_scale': 1e-3, 'temperature': 1.0})
        train_info = self.dataset('train')
        validation_info = self.dataset('validation')
        return SAGNTrainer(self.config, self.args, model, train_info,
                           validation=validation_info)

    def test_offline_training_builds_a_pool_and_calibrates_scores(self):
        trainer = self.trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
        self.assertTrue((trainer.results_folder / 'checkpoint-best.pt').is_file())
        self.assertTrue(trainer.pool)
        self.assertLessEqual(len(trainer.pool), 2)
        self.assertTrue(trainer.model.get_calibration()['calibrated'])
        checkpoint = torch.load(trainer.results_folder / 'checkpoint-best.pt',
                                map_location='cpu')
        self.assertEqual(checkpoint['protocol'], trainer.protocol())
        self.assertIn('calibration', checkpoint)
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.load('best')

    def test_online_evaluation_writes_reports_and_isolates_validation(self):
        trainer = self.trainer()
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train()
            trainer.load('best')
            validation = self.dataset('validation')
            test = self.dataset('test')
            initial_pool = deepcopy(trainer.pool)
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            expected_random = (random.random(), np.random.rand(), torch.rand(1))
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            with trainer.validation_prediction_context():
                trainer.restore(validation['dataloader'], sampling_steps=1)
            self.assertEqual(len(trainer.pool), len(initial_pool))
            for restored, original in zip(trainer.pool, initial_pool):
                self.assertEqual(restored['step'], original['step'])
                self.assertEqual(restored['loss'], original['loss'])
                self.assertEqual(restored['state'].keys(), original['state'].keys())
                for key, value in original['state'].items():
                    torch.testing.assert_close(restored['state'][key], value)
            self.assertEqual(random.random(), expected_random[0])
            self.assertEqual(np.random.rand(), expected_random[1])
            torch.testing.assert_close(torch.rand(1), expected_random[2])
            selected = trainer.decision_threshold
            self.assertIsNotNone(
                trainer.last_detection_report['validation_threshold_selection'])
            samples = trainer.restore(test['dataloader'], sampling_steps=1)
        self.assertEqual(samples.shape, (len(test['dataset']), 10, 3))
        self.assertEqual(trainer.last_detection_report['threshold'], selected)
        self.assertIsNone(
            trainer.last_detection_report['validation_threshold_selection'])
        self.assertTrue(np.isfinite(samples).all())
        output = Path(self.args.save_dir)
        self.assertTrue((output / 'toy_sagn_online_test.json').is_file())
        online_log = json.loads(
            (output / 'toy_sagn_online_test.json').read_text(encoding='utf-8'))
        self.assertTrue(online_log['selections'])
        selected_metrics = pd.read_csv(output / 'toy_selected_threshold_metrics.csv')
        self.assertEqual(selected_metrics['split'].tolist(), ['validation', 'test'])
        self.assertEqual(selected_metrics['threshold'].nunique(), 1)

    def test_residual_score_uses_only_observed_targets(self):
        trainer = self.trainer()
        trainer.model.eval()
        buffer = deque(maxlen=10)
        buffer.append((100, np.array([1.0, 1.0, 1.0])))
        buffer.append((101, np.array([5.0, 5.0, 5.0])))
        before = trainer._residual_score_for_window(buffer, 99)
        after = trainer._residual_score_for_window(buffer, 101)
        self.assertEqual(before, 0.0)
        self.assertGreaterEqual(after, 0.0)

    def test_restore_requires_the_train_only_scaling_protocol(self):
        trainer = self.trainer()
        config = deepcopy(self.config)
        config['dataloader']['params'] = dict(
            config['dataloader']['params'], feature_scaling='legacy',
            period='test')
        with self.assertRaisesRegex(ValueError, 'train_minmax'):
            info = build_dataloader_cond(config, self.args)
            trainer.restore(info['dataloader'])


if __name__ == '__main__':
    unittest.main()
