import unittest

import torch
from torch import nn

from Models.interpretable_diffusion.gaussian_diffusion import Diffusion_TS


class FixedGranger(nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores
        self.last_full_error = torch.zeros_like(scores)

    def forward(self, metrics, rps):
        return self.scores.sum() * 0, self.scores


class FixedAttention(nn.Module):
    def forward(self, metrics, rps):
        return metrics + 2, torch.ones_like(metrics)


class IdentityEncoder(nn.Module):
    def forward(self, condition, step):
        return condition


class RecordingAdapter(nn.Module):
    def forward(self, history, error, scores, update_memory=True):
        self.scores = scores.clone()
        return history.new_zeros(len(history), 1)


class CausalMaskThresholdTests(unittest.TestCase):
    def prepare(self, threshold=None):
        scores = torch.tensor([[-0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.31]], requires_grad=True)
        history = torch.randn(1, 20, 9)
        diffusion = Diffusion_TS(
            seq_length=30, feature_size=10, DeltaT=10, d_model=16,
            n_layer_enc=1, n_layer_dec=1, timesteps=3,
            drift_adaptation={'enabled': True}, causal_mask_threshold=threshold,
        )
        model = diffusion.model
        model.atte1 = FixedAttention()
        model.atte2 = FixedAttention()
        model.emb_cond = nn.Identity()
        model.pos_enc = nn.Identity()
        model.encoder = IdentityEncoder()
        model.drift_adapter = RecordingAdapter()
        context = model.prepare_history(history, FixedGranger(scores))
        return model, context, history, scores

    def test_sweep_masks_condition_and_adapter_scores_with_strict_comparison(self):
        for threshold in (0, 0.05, 0.1, 0.15, 0.2, 0.3):
            with self.subTest(threshold=threshold):
                model, context, history, scores = self.prepare(threshold)
                expected = torch.where(scores.detach() > threshold, scores.detach(), 0)
                torch.testing.assert_close(model.drift_adapter.scores, expected)
                torch.testing.assert_close(context['condition_embedding'], history[:, :, 1:] + 2 * expected.unsqueeze(1))
                self.assertFalse(model.drift_adapter.scores.requires_grad)
                self.assertTrue(model.history_adaptation_enabled)
                self.assertEqual(model.causal_mask_threshold, threshold)

    def test_omitting_threshold_preserves_unmasked_history_path(self):
        model, context, history, scores = self.prepare()
        torch.testing.assert_close(model.drift_adapter.scores, scores.detach())
        torch.testing.assert_close(context['condition_embedding'], history[:, :, 1:] + 2 * scores.detach().unsqueeze(1))
        self.assertEqual(model.causal_mask_threshold, 0.1)
        self.assertFalse(model.mask_history_scores)

    def test_invalid_threshold_is_rejected(self):
        for threshold in (-0.01, 1.01, float('nan'), float('inf')):
            with self.subTest(threshold=threshold):
                with self.assertRaisesRegex(ValueError, 'causal_mask_threshold'):
                    self.prepare(threshold)


if __name__ == '__main__':
    unittest.main()
