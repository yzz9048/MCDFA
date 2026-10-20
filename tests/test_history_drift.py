import contextlib
import io
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from ema_pytorch import EMA

from Models.interpretable_diffusion.history_drift import HistoryDriftAdapter
from Models.interpretable_diffusion.model_utils import Linear
from Models.interpretable_diffusion.Granger import GrangerNet
from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS
from Utils.Data_utils.real_datasets import CustomDataset
from Utils.detection_report import select_validation_threshold
from Utils.io_utils import load_yaml_config
from engine.solver import Trainer


class HistoryDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(42)

    def controller(self, **kwargs):
        return HistoryDriftAdapter(3, 10, 16, memory_warmup=2,
                                   confirmation_windows=2, **kwargs)

    def histories(self, count=8):
        time = torch.arange(20).float()[None, :, None]
        level = torch.arange(count).float()[:, None, None] * 0.02
        return (time * 0.005 + level).expand(count, 20, 3).clone()

    def model(self, steps=3):
        return Diffusion_TS(seq_length=30, feature_size=4, DeltaT=10, d_model=16,
                            timesteps=steps, n_layer_enc=1, n_layer_dec=1,
                            drift_adaptation={'enabled': True, 'memory_warmup': 2},
                            clip_denoised=False)

    def test_frozen_ema_runs_the_same_projection_and_adaptation(self):
        layer = Linear(2, 16, 10, resid_pdrop=0, consistent_inference=True).eval()
        ema = EMA(layer, beta=0.99)
        x = torch.randn(2, 10, 16)
        neutral = torch.zeros(2, 36)
        changed = torch.randn(2, 36) * 0.1
        outputs = []
        for adaptation in (neutral, changed):
            layer.assign_adaptation(adaptation)
            ema.ema_model.assign_adaptation(adaptation)
            torch.testing.assert_close(layer(x), ema.ema_model(x))
            outputs.append(ema.ema_model(x))
        self.assertGreater(float((outputs[0] - outputs[1]).abs().max()), 1e-3)

    def test_both_granger_predictors_and_load_attention_receive_gradients(self):
        granger = GrangerNet(2, 20, hidden_dim=16, joint_training=True)
        loss, _ = granger(torch.randn(3, 20, 2), torch.randn(3, 20, 1))
        loss.backward()
        for module in (granger.restricted_predictor, granger.full_predictor, granger.cross_attention):
            self.assertTrue(all(parameter.grad is not None for parameter in module.parameters()))

    def test_batch_partitioning_does_not_change_adaptation_or_memory(self):
        whole, split = self.controller(), self.controller()
        split.load_state_dict(whole.state_dict())
        history = self.histories()
        residual = torch.full((8, 2), 0.01)
        scores = torch.full((8, 2), 0.8)
        expected = whole(history, residual, scores)
        actual = torch.cat([split(history[i:i+1], residual[i:i+1], scores[i:i+1]) for i in range(8)])
        torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
        for name in ('fast', 'slow', 'candidate'):
            np.testing.assert_array_equal(whole.memory[name], split.memory[name])
        self.assertEqual(whole.memory['observations'], 8)

    def test_later_observations_cannot_change_an_earlier_adaptation(self):
        first, second = self.controller(), self.controller()
        second.load_state_dict(first.state_dict())
        history = self.histories()
        changed = history.clone()
        changed[4:] += 10
        residual = torch.full((8, 2), 0.01)
        scores = torch.full((8, 2), 0.8)
        a = first(history, residual, scores)
        b = second(changed, residual, scores)
        torch.testing.assert_close(a[:4], b[:4])

    def test_zero_drift_is_identity_and_large_residual_cannot_poison_memory(self):
        controller = self.controller()
        history = self.histories(1)
        residual, scores = torch.full((1, 2), 0.01), torch.ones(1, 2)
        controller(history, residual, scores)
        identity = controller(history, residual, scores)
        torch.testing.assert_close(identity, torch.zeros_like(identity))
        before = controller.get_state()
        adaptation = controller(history + 3, residual * 100, scores)
        np.testing.assert_array_equal(controller.memory['fast'], before['fast'])
        np.testing.assert_array_equal(controller.memory['slow'], before['slow'])
        self.assertFalse(controller.last_diagnostics[0]['memory_updated'])
        self.assertTrue(torch.isfinite(adaptation).all())
        self.assertLessEqual(float(adaptation.abs().max()), 0.15)

    def test_memory_ablation_keeps_a_local_adaptation_without_persistent_state(self):
        controller = self.controller(use_memory=False)
        adaptation = controller(self.histories(3), torch.full((3, 2), .01), torch.ones(3, 2))
        self.assertGreater(float(adaptation.abs().sum()), 0)
        self.assertEqual(controller.memory['observations'], 0)

    def test_disabling_load_gate_keeps_residual_protection(self):
        controller = self.controller(use_load_gate=False)
        residual, scores = torch.full((1, 2), .01), torch.ones(1, 2)
        history = self.histories(1)
        controller(history, residual, scores)
        before = controller.get_state()
        adaptation = controller(history + 2, residual * 100, scores)
        self.assertLess(float(adaptation.abs().max()), 1e-6)
        np.testing.assert_array_equal(before['slow'], controller.memory['slow'])

    def test_diffusion_reuses_one_history_and_does_not_backpropagate_at_inference(self):
        model, granger = self.model().eval(), GrangerNet(2, 20, hidden_dim=16, joint_training=True).eval()
        history = self.histories(3)
        with patch.object(granger, 'forward', wraps=granger.forward) as call:
            with contextlib.redirect_stderr(io.StringIO()):
                sample, logits, _ = model.sample_infill(granger, 10, [3, 10, 2], history, True, None, None)
        self.assertEqual(call.call_count, 1)
        self.assertEqual(model.model.drift_adapter.memory['observations'], 3)
        self.assertEqual(tuple(sample.shape), (3, 10, 2))
        self.assertEqual(tuple(logits.shape), (3, 2))
        self.assertTrue(all(parameter.grad is None for parameter in granger.parameters()))

    def test_load_difference_is_along_time_and_independent_of_other_samples(self):
        model, granger = self.model().eval(), GrangerNet(2, 20, hidden_dim=16, joint_training=True).eval()
        history = self.histories(3)
        observed = []
        hook = model.model.atte2.register_forward_pre_hook(lambda module, args: observed.append(args[1].clone()))
        with torch.no_grad():
            model.model.prepare_history(history, granger)
        hook.remove()
        expected = torch.cat([torch.zeros_like(history[:, :1, :1]), torch.diff(history[:, :, :1], dim=1)], dim=1)
        torch.testing.assert_close(observed[0], expected)

    def test_sampling_safety_bounds_preserve_out_of_training_range_amplitudes(self):
        model = self.model()
        forecast = torch.full((1, 10, 2), 10.)
        context = {'observed_metric_scale': torch.tensor([[[5., 1.]]])}
        with patch.object(model, 'model_predictions', return_value=(None, forecast, torch.zeros(1, 2), torch.tensor(0.))):
            _, _, _, start, _, _ = model.p_mean_variance(None, torch.zeros_like(forecast), None,
                torch.zeros(1, dtype=torch.long), True, clip_denoised=False, history_context=context)
        torch.testing.assert_close(start[:, :, 0], torch.full((1, 10), 10.))
        torch.testing.assert_close(start[:, :, 1], torch.full((1, 10), 3.))

    def test_scaling_uses_only_training_data_and_forecasts_invert_exactly(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'toy.csv'
            frame = pd.DataFrame({'timestamp': np.arange(201), 'rps': np.arange(201) + 10,
                                  'a': np.arange(201) + 100., 'b': np.arange(201) * 2.,
                                  'label': np.zeros(201)})
            frame.loc[161:, 'a'] += 10000
            frame.to_csv(path, index=False)
            kwargs = dict(name='toy', data_root=str(path), window=30, DeltaT=1, proportion=0.6,
                          validation_proportion=0.2, forecast_length=10, feature_scaling='train_minmax',
                          save2npy=False, output_dir=folder)
            train = CustomDataset(period='train', **kwargs)
            test = CustomDataset(period='test', predict_length=10, **kwargs)
            self.assertEqual(train.normalization_state(), test.normalization_state())
            self.assertEqual(train.scaler.data_max_[1], 220)
            self.assertGreater(test.rawdata[-1, 1], 1)
            restored = test.inverse_prediction(test.rawdata[-10:, 1:-1][None])
            np.testing.assert_allclose(restored[0], frame.iloc[-10:, 2:-1].to_numpy())
            np.testing.assert_allclose(test.normalize(test.unnormalize(test.samples[:2])), test.samples[:2])

    def test_threshold_prefers_more_hits_even_with_worse_event_precision(self):
        labels = np.zeros(45)
        labels[15:17] = labels[30:32] = 1
        rows = [{'split': 'validation', 'window_start': i, 'observation_end': i,
                 'observation_timestamp': str(i),
                 'target_end': i+3, 'window_label': int(labels[i+1:i+4].any()),
                 'fault_probability': 0.1} for i in range(40)]
        for index, probability in ((2, .6), (5, .6), (12, .6), (27, .9)):
            rows[index]['fault_probability'] = probability
        selection, sweep = select_validation_threshold(rows, labels, np.arange(45),
            {'threshold_selection_objective': 'hits_lead', 'decision_threshold': 0.5})
        chosen = selection['selected_validation_metrics']
        self.assertEqual(chosen['matched_alarm_events'], 2)
        self.assertEqual(chosen['total_matched_lead_time_timestamps'], 6)
        self.assertLess(chosen['event_precision'], max(row['event_precision'] for row in sweep))

    def test_gaia_stays_on_legacy_model_and_data_protocol(self):
        config = load_yaml_config('config/predict.yaml')
        self.assertFalse(config['model']['params'].get('drift_adaptation', {}).get('enabled', False))
        self.assertNotIn('feature_scaling', config['dataloader']['params'])

    def test_validation_restores_drift_memory_and_future_values_do_not_affect_scores(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'toy.csv'
            pd.DataFrame({'time': np.arange(201), 'rps': np.sin(np.arange(201)/20)+2,
                          'a': np.arange(201)/100, 'b': np.arange(201)/200,
                          'label': (np.arange(201) == 184).astype(int)}).to_csv(path, index=False)
            kwargs = dict(name='toy', data_root=str(path), window=30, DeltaT=1, proportion=0.6,
                          validation_proportion=0.2, forecast_length=10, predict_length=10,
                          feature_scaling='train_minmax', save2npy=False, output_dir=folder)
            test, validation = CustomDataset(period='test', **kwargs), CustomDataset(period='validation', **kwargs)
            model = self.model(steps=2)
            granger = GrangerNet(2, 20, hidden_dim=16, joint_training=True)
            config = {'model': {'params': {'drift_adaptation': {'enabled': True}}},
                      'dataloader': {'params': {'proportion': .6, 'validation_proportion': .2, 'DeltaT': 1},
                                     'threshold_selection_objective': 'hits_lead', 'drift_warmup_windows': 4},
                      'solver': {'max_epochs': 1, 'gradient_accumulate_every': 1, 'save_cycle': 1,
                                 'results_folder': str(root/'checkpoints'), 'base_lr': 1e-3,
                                 'ema': {'decay': .9, 'update_interval': 1},
                                 'scheduler': {'target': 'engine.lr_sch.ReduceLROnPlateauWithWarmup',
                                               'params': {'warmup': 5, 'warmup_lr': .01, 'min_lr': 0}}}}
            args = SimpleNamespace(pred_len=10, clip=True, save_dir=str(root/'reports'))
            trainer = Trainer(config, args, model, granger, {'dataset': test, 'dataloader': DataLoader(test, batch_size=16)})
            with torch.no_grad():
                trainer.ema.ema_model.model.prepare_history(torch.tensor(test.samples[:2, :20], dtype=torch.float32),
                                                           trainer.emaG.ema_model)
            initial = trainer.ema.ema_model.model.drift_adapter.get_state()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with trainer.validation_prediction_context():
                    trainer.restore(DataLoader(validation, batch_size=16), sampling_steps=2)
                    self.assertGreater(trainer.ema.ema_model.model.drift_adapter.memory['observations'], 0)
            restored = trainer.ema.ema_model.model.drift_adapter.get_state()
            for name in ('fast', 'slow', 'candidate'):
                np.testing.assert_array_equal(initial[name], restored[name])
            self.assertEqual(initial['observations'], restored['observations'])
            self.assertFalse(trainer.optG.state)
            # The final window contains future data that cannot appear in any input history.
            rng = torch.get_rng_state()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                original = trainer.restore(DataLoader(test, batch_size=16), sampling_steps=2)
            scores = pd.read_csv(root/'reports'/'toy_window_scores_test.csv')['fault_probability'].to_numpy()
            test.samples[-1, -1, :] += 10000
            torch.set_rng_state(rng)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                changed = trainer.restore(DataLoader(test, batch_size=16), sampling_steps=2)
            np.testing.assert_array_equal(original, changed)
            np.testing.assert_array_equal(scores, pd.read_csv(root/'reports'/'toy_window_scores_test.csv')['fault_probability'])
            trainer.save('best')
            trainer.load('best')
            self.assertTrue((root/'reports'/'toy_drift_diagnostics_test.csv').is_file())
            config['dataloader']['online_granger_update'] = True
            with patch.object(granger, 'forward', wraps=granger.forward) as calls:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    trainer.restore(DataLoader(test, batch_size=16), sampling_steps=2)
            self.assertGreater(calls.call_count, 0)
            self.assertLessEqual(calls.call_count, len(test))
            self.assertTrue(trainer.optG.state)


if __name__ == '__main__':
    unittest.main()
