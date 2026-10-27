"""History-only adaptation with chronological, label-free background memory."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class HistoryDriftAdapter(nn.Module):
    def __init__(self, feature_count, horizon, embedding_dim, **config):
        super().__init__()
        self.feature_count = feature_count
        self.horizon = horizon
        self.embedding_dim = embedding_dim
        self.fast_rate = float(config.get('fast_rate', 0.2))
        self.slow_rate = float(config.get('slow_rate', 0.02))
        self.warmup = int(config.get('memory_warmup', 16))
        self.confirmation = int(config.get('confirmation_windows', 8))
        self.residual_limit = float(config.get('residual_ratio_limit', 3.0))
        self.max_scale = float(config.get('max_scale', 0.15))
        self.max_shift = float(config.get('max_shift', 0.1))
        self.enabled = bool(config.get('enable_adapter', True))
        self.use_load_gate = bool(config.get('use_load_gate', True))
        self.use_memory = bool(config.get('use_memory', True))
        if not 0 < self.slow_rate <= self.fast_rate <= 1:
            raise ValueError('Require 0 < slow_rate <= fast_rate <= 1.')
        if self.warmup < 1 or self.confirmation < 1 or self.residual_limit <= 1:
            raise ValueError('Invalid drift memory or residual configuration.')
        if not 0 <= self.max_scale < 1 or self.max_shift < 0:
            raise ValueError('Adapter scales must stay positive and shifts bounded.')
        concept_dim = int(config.get('concept_dim', 64))
        hidden_dim = int(config.get('bottleneck_dim', 32))
        # Store raw descriptors, so changing encoder weights cannot invalidate memory.
        self.encoder = nn.Sequential(nn.Linear(feature_count * 4, concept_dim),
                                     nn.GELU(), nn.Linear(concept_dim, concept_dim))
        self.generator = nn.Sequential(nn.Linear(concept_dim + 2, hidden_dim),
                                       nn.SiLU(), nn.Linear(hidden_dim, 2 * horizon + embedding_dim))
        nn.init.normal_(self.generator[-1].weight, std=1e-3)
        nn.init.zeros_(self.generator[-1].bias)
        # Runtime state is deliberately excluded from model/EMA parameter averaging.
        self.reset_state()
        self.last_penalty = None
        self.last_diagnostics = []

    def reset_state(self):
        self.memory = {'fast': None, 'slow': None, 'candidate': None,
                       'candidate_count': 0, 'residual': None, 'observations': 0}

    def get_state(self):
        return deepcopy(self.memory)

    def set_state(self, state):
        self.memory = deepcopy(state)

    @staticmethod
    def descriptors(history):
        mean = history.mean(dim=1)
        std = history.std(dim=1, unbiased=False)
        slope = (history[:, -1] - history[:, 0]) / max(history.shape[1] - 1, 1)
        last_offset = history[:, -1] - mean
        return torch.stack([mean, std, slope, last_offset], dim=-1).flatten(1)

    @staticmethod
    def _rms(values):
        return float(np.sqrt(np.mean(np.square(values))))

    def forward(self, history, residual_error, causal_scores, update_memory=True):
        stats = self.descriptors(history)
        values = stats.detach().cpu().numpy()
        local_references = self.descriptors(history[:, :max(2, history.shape[1] // 2)]).detach().cpu().numpy()
        errors = residual_error.detach().mean(dim=-1).cpu().numpy()
        reliability = causal_scores.detach().mean(dim=-1).cpu().numpy()
        references, strengths, speeds, gates, diagnostics = [], [], [], [], []
        for current, local_reference, error, causal in zip(values, local_references, errors, reliability):
            state = self.memory if self.use_memory else {
                'fast': local_reference, 'slow': local_reference, 'candidate': None,
                'candidate_count': 0, 'residual': max(float(error), 1e-4), 'observations': self.warmup}
            if state['slow'] is None:
                reference = current.copy()
                ratio, strength, speed, load_gate, residual_gate = 1.0, 0.0, 0.0, 0.0, 1.0
                admitted = True
            else:
                reference = state['slow'].copy()
                delta = (current - reference).reshape(self.feature_count, 4)
                metric_change = self._rms(delta[1:])
                load_change = self._rms(delta[0])
                strength = float(np.tanh(self._rms(delta) / (0.1 + self._rms(reference))))
                speed = float(np.tanh(self._rms(state['fast'] - reference)))
                ratio = float(error / max(state['residual'], 1e-4))
                residual_gate = float(np.exp(-max(ratio - 1.0, 0.0)))
                load_gate = min(1.0, load_change / (metric_change + 0.02))
                load_gate *= 0.25 + 0.75 * float(np.clip(causal, 0, 1))
                # Small normal changes may be adapted even when RPS is stationary.
                background_gate = 0.25 if metric_change <= 0.05 and ratio <= 1.5 else 0.0
                load_gate = max(load_gate, background_gate)
                admitted = ratio <= self.residual_limit and (
                    not self.use_load_gate or metric_change <= 0.08 or
                    (load_change >= 0.03 and load_gate >= 0.25))
                if state['observations'] < self.warmup:
                    admitted = ratio <= self.residual_limit
            gate = residual_gate * (load_gate if self.use_load_gate else 1.0)
            references.append(reference)
            strengths.append(strength)
            speeds.append(speed)
            gates.append(gate)
            if update_memory and self.use_memory:
                if state['slow'] is None:
                    state['fast'] = current.copy()
                    state['slow'] = current.copy()
                    state['residual'] = max(float(error), 1e-4)
                elif admitted:
                    state['fast'] += self.fast_rate * (current - state['fast'])
                    if state['candidate'] is not None and self._rms(current - state['candidate']) <= 0.05:
                        state['candidate_count'] += 1
                        state['candidate'] += self.fast_rate * (current - state['candidate'])
                    else:
                        state['candidate'], state['candidate_count'] = current.copy(), 1
                    if state['candidate_count'] >= self.confirmation or state['observations'] < self.warmup:
                        state['slow'] += self.slow_rate * (current - state['slow'])
                    state['residual'] += self.slow_rate * (float(error) - state['residual'])
                else:
                    state['candidate'], state['candidate_count'] = None, 0
                state['observations'] += 1
            diagnostics.append({'drift_strength': strength, 'fast_slow_difference': speed,
                                'load_gate': load_gate, 'residual_gate': residual_gate,
                                'adaptation_gate': gate, 'residual_ratio': ratio,
                                'memory_updated': bool(update_memory and self.use_memory and admitted),
                                'history_residual': float(error)})
        reference = stats.new_tensor(np.stack(references))
        direction = F.normalize(self.encoder(stats) - self.encoder(reference), dim=-1, eps=1e-6)
        strength = stats.new_tensor(strengths).unsqueeze(1)
        generator_input = torch.cat([direction, strength, stats.new_tensor(speeds).unsqueeze(1)], dim=1)
        response = self.generator(generator_input) - self.generator(torch.zeros_like(generator_input))
        bounds = stats.new_full((2 * self.horizon + self.embedding_dim,), self.max_scale)
        bounds[-self.horizon:] = self.max_shift
        adaptation = bounds * torch.tanh(response) * strength * stats.new_tensor(gates).unsqueeze(1)
        if not self.enabled:
            adaptation = adaptation * 0
        self.last_penalty = adaptation.square().mean()
        amplitudes = adaptation.detach().abs().mean(dim=1).cpu().tolist()
        for row, amplitude in zip(diagnostics, amplitudes):
            row['adaptation_mean_abs'] = amplitude
        self.last_diagnostics = diagnostics
        return adaptation
