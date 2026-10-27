"""SAGN predictor for multi-stream concept drift.

This module follows Zhou et al., "Multi-Stream Concept Drift Self-Adaptation
Using Graph Neural Network" (IEEE TKDE):

* the correlation graph is sampled with the Gumbel-Softmax trick (Eq. 2--5);
* the predictor is a sequence-to-sequence diffusion-convolutional GRU
  (DCRNN cell, Eq. after Eq. 5) conditioned on that graph;
* graph and network parameters are updated online from newly observed samples
  (handled by ``engine.sagn_solver.SAGNTrainer``);
* the GNN pool / dynamic validation set selection is handled by the trainer.

Repository-specific mapping (agreed with the user):

* every service metric is one stream/node,
* node features are ``[service metric, RPS]`` (RPS is shared by every node),
* the target series of a node is its service metric over the forecast horizon.

The offline model predicts normalized values, exactly like the previous
diffusion predictor, so the existing data split, forecast horizon and detection
reporting contracts stay unchanged.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F


def _as_dict(value):
    return dict(value or {})


def _gumbel_noise(shape, device):
    uniform = torch.rand(shape, device=device).clamp_(1e-20, 1.0 - 1e-20)
    return -torch.log(-torch.log(uniform))


class NodeFeatureExtractor(nn.Module):
    """Feature extraction network ``Fetc`` from Section 4.1.1.

    Two convolutional layers, two normalization layers and a fully connected
    layer are applied per node, then the temporal representation is pooled.
    """

    def __init__(self, feature_count, conv_dim=32, embed_dim=32):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_count, conv_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(conv_dim)
        self.conv2 = nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(conv_dim)
        self.fc = nn.Linear(conv_dim, embed_dim)
        self.activation = nn.GELU()

    def forward(self, node_input):
        # node_input: (batch, history, nodes, features)
        batch, steps, nodes, features = node_input.shape
        x = node_input.permute(0, 2, 3, 1).reshape(batch * nodes, features, steps)
        x = self.activation(self.norm1(self.conv1(x).transpose(1, 2)).transpose(1, 2))
        x = self.activation(self.norm2(self.conv2(x).transpose(1, 2)).transpose(1, 2))
        x = x.mean(dim=-1)
        return self.fc(x).view(batch, nodes, -1)


class GumbelCorrelationGraph(nn.Module):
    """Sampled correlation graph structure (Section 4.1.1, Eq. 5).

    For every ordered pair of nodes ``(u, v)`` the pair representation yields a
    Bernoulli logit ``log(mu / (1 - mu))``.  Gumbel noise is applied at node
    level, as in ``Gum_u - Gum_v``, and a temperature-scaled logistic map gives
    the continuous edge weight used by the diffusion convolution.
    """

    def __init__(self, feature_count=2, conv_dim=32, embed_dim=32,
                 temperature=0.5, temperature_min=0.1, eval_temperature=None):
        super().__init__()
        if temperature <= 0 or temperature_min <= 0:
            raise ValueError('Gumbel temperatures must be positive.')
        self.extractor = NodeFeatureExtractor(feature_count, conv_dim, embed_dim)
        self.pair = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )
        self.register_buffer('temperature', torch.tensor(float(temperature)))
        self.temperature_min = float(temperature_min)
        self.eval_temperature = float(
            temperature_min if eval_temperature is None else eval_temperature
        )
        if self.eval_temperature <= 0:
            raise ValueError('eval_temperature must be positive.')

    def set_temperature(self, value):
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Graph temperature must be finite and positive.')
        self.temperature.fill_(value)

    def current_temperature(self, training):
        return float(self.temperature) if training else self.eval_temperature

    def logits(self, node_input):
        embeddings = self.extractor(node_input)  # (batch, nodes, embed)
        batch, nodes, _ = embeddings.shape
        left = embeddings.unsqueeze(2).expand(batch, nodes, nodes, -1)
        right = embeddings.unsqueeze(1).expand(batch, nodes, nodes, -1)
        return self.pair(torch.cat([left, right], dim=-1)).squeeze(-1)

    def forward(self, node_input, sample=None, temperature=None):
        raw = self.logits(node_input)
        if sample is None:
            sample = self.training
        if temperature is None:
            temperature = self.current_temperature(sample)
        temperature = max(float(temperature), 1e-4)
        if sample:
            batch, nodes = raw.shape[:2]
            noise = _gumbel_noise((batch, nodes), raw.device)
            pair_noise = noise.unsqueeze(2) - noise.unsqueeze(1)
            graph = torch.sigmoid((raw + pair_noise) / temperature)
        else:
            graph = torch.sigmoid(raw / temperature)
        return graph, raw


class DiffusionConv(nn.Module):
    """Graph diffusion convolution ``W_lambda *_G [X || H]``.

    ``sum_k w_{k,1} (D_out^-1 G)^k X + w_{k,2} (D_in^-1 G^T)^k X``
    as defined for DCRNN and used by SAGN.
    """

    def __init__(self, input_dim, output_dim, steps=2):
        super().__init__()
        if steps < 0:
            raise ValueError('Diffusion steps must be non-negative.')
        self.steps = int(steps)
        self.weights = nn.Parameter(torch.empty(2, self.steps + 1, input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    @staticmethod
    def normalized_adjacency(graph):
        # graph: (batch, nodes, nodes)
        # D_out^-1 G and D_in^-1 G^T are both row-stochastic.
        out_degree = graph.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        in_degree = graph.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        forward = graph / out_degree
        backward = graph.transpose(1, 2) / in_degree.transpose(1, 2)
        return forward, backward

    def forward(self, x, graph):
        forward, backward = self.normalized_adjacency(graph)
        forward_state, backward_state = x, x
        output = x.new_zeros(x.shape[0], x.shape[1], self.weights.shape[-1])
        for order in range(self.steps + 1):
            if order > 0:
                forward_state = torch.bmm(forward, forward_state)
                backward_state = torch.bmm(backward, backward_state)
            output = output + torch.einsum('bsi,io->bso', forward_state,
                                           self.weights[0, order])
            output = output + torch.einsum('bsi,io->bso', backward_state,
                                           self.weights[1, order])
        return output + self.bias


class DiffusionConvGRUCell(nn.Module):
    """Diffusion convolutional GRU cell from DCRNN.

    ``r = sigmoid(W_r *_G [X || H])``
    ``q = tanh(W_q *_G [X || (r * H)])``
    ``c = sigmoid(W_c *_G [X || H])``
    ``H' = c * H + (1 - c) * q``
    """

    def __init__(self, input_dim, hidden_dim, steps=2):
        super().__init__()
        self.reset_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, steps)
        self.update_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, steps)
        self.candidate_conv = DiffusionConv(input_dim + hidden_dim, hidden_dim, steps)

    def forward(self, x, hidden, graph):
        reset = torch.sigmoid(self.reset_conv(torch.cat([x, hidden], dim=-1), graph))
        candidate = torch.tanh(
            self.candidate_conv(torch.cat([x, reset * hidden], dim=-1), graph)
        )
        update = torch.sigmoid(self.update_conv(torch.cat([x, hidden], dim=-1), graph))
        return update * hidden + (1.0 - update) * candidate


class SAGN(nn.Module):
    """Sequence-to-sequence GNN predictor described by SAGN."""

    def __init__(
        self,
        seq_length,
        feature_size,
        DeltaT,
        hidden_dim=64,
        graph_conv_dim=32,
        graph_embed_dim=32,
        graph_steps=2,
        gumbel_temperature=0.5,
        gumbel_temperature_min=0.1,
        gumbel_anneal_steps=None,
        eval_temperature=None,
        graph_l1=0.0,
        teacher_forcing=1.0,
        teacher_forcing_decay_steps=None,
        residual_output=True,
        node_feature_count=2,
        pool=None,
        online=None,
        residual_score=None,
        **kwargs,
    ):
        super().__init__()
        if feature_size < 3:
            raise ValueError('feature_size must include RPS, at least one service and a label.')
        if not 0 < DeltaT < seq_length:
            raise ValueError('DeltaT must be a forecast horizon inside seq_length.')
        self.seq_length = int(seq_length)
        self.DeltaT = int(DeltaT)
        self.feature_size = int(feature_size)
        self.history_length = self.seq_length - self.DeltaT
        self.node_count = self.feature_size - 2
        self.node_feature_count = int(node_feature_count)
        if self.node_feature_count != 2:
            raise ValueError('SAGN uses node features [service metric, RPS].')
        self.hidden_dim = int(hidden_dim)
        self.graph_steps = int(graph_steps)
        self.graph_l1 = float(graph_l1)
        if self.graph_l1 < 0:
            raise ValueError('graph_l1 must be non-negative.')
        self.teacher_forcing = float(teacher_forcing)
        if not 0.0 <= self.teacher_forcing <= 1.0:
            raise ValueError('teacher_forcing must be in [0, 1].')
        self.teacher_forcing_decay_steps = (
            None if teacher_forcing_decay_steps is None
            else int(teacher_forcing_decay_steps)
        )
        self.residual_output = bool(residual_output)
        self.gumbel_anneal_steps = (
            None if gumbel_anneal_steps is None else int(gumbel_anneal_steps)
        )
        self.gumbel_temperature_init = float(gumbel_temperature)
        self.gumbel_temperature_min = float(gumbel_temperature_min)

        self.graph = GumbelCorrelationGraph(
            feature_count=self.node_feature_count,
            conv_dim=graph_conv_dim,
            embed_dim=graph_embed_dim,
            temperature=gumbel_temperature,
            temperature_min=gumbel_temperature_min,
            eval_temperature=eval_temperature,
        )
        self.encoder_cell = DiffusionConvGRUCell(
            self.node_feature_count, self.hidden_dim, self.graph_steps
        )
        self.decoder_cell = DiffusionConvGRUCell(1, self.hidden_dim, self.graph_steps)
        self.hidden_init = nn.Parameter(
            torch.zeros(1, self.node_count, self.hidden_dim)
        )
        self.output_head = nn.Linear(self.hidden_dim, 1)
        if self.residual_output:
            # Start from persistence; the decoder learns level corrections.
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)

        self.pool_config = _as_dict(pool)
        self.online_config = _as_dict(online)
        self.residual_config = _as_dict(residual_score)
        self.score_aggregation = str(
            self.residual_config.get('aggregation', 'max')
        ).lower()
        if self.score_aggregation not in {'max', 'mean', 'topk'}:
            raise ValueError('residual_score.aggregation must be max, mean or topk.')
        self.score_top_k = int(self.residual_config.get('top_k', 3))
        self.score_temperature = float(self.residual_config.get('temperature', 1.0))
        self.score_min_scale = float(self.residual_config.get('min_scale', 1e-3))
        if self.score_temperature <= 0 or self.score_min_scale <= 0:
            raise ValueError('Residual score temperature and min_scale must be positive.')
        self.register_buffer('residual_center', torch.zeros(self.node_count))
        self.register_buffer('residual_scale', torch.ones(self.node_count))
        self.register_buffer('residual_aggregate_scale', torch.ones(1))
        self.register_buffer('residual_calibrated', torch.zeros(1, dtype=torch.bool))

    # ------------------------------------------------------------------
    # graph / temperature
    # ------------------------------------------------------------------
    def set_graph_temperature(self, value):
        self.graph.set_temperature(value)

    def annealed_temperature(self, step):
        if not self.gumbel_anneal_steps or self.gumbel_anneal_steps <= 0:
            return self.gumbel_temperature_init
        ratio = min(max(float(step), 0.0) / self.gumbel_anneal_steps, 1.0)
        low, high = self.gumbel_temperature_min, self.gumbel_temperature_init
        if low <= 0:
            return high
        return high * (low / high) ** ratio

    def teacher_forcing_probability(self, step):
        """Scheduled sampling: match training with autoregressive inference."""
        if not self.teacher_forcing_decay_steps:
            return self.teacher_forcing
        ratio = min(max(float(step), 0.0) / self.teacher_forcing_decay_steps, 1.0)
        return self.teacher_forcing * (1.0 - ratio)

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def prepare_nodes(self, history):
        """Convert ``(batch, history, rps + metrics)`` to per-node features.

        Returns node input ``(batch, history, nodes, 2)`` and metric history
        ``(batch, history, nodes)``.
        """
        expected = self.node_count + 1
        if history.dim() != 3 or history.shape[-1] != expected:
            raise ValueError(
                f'history must have shape (batch, history, {expected}), '
                f'got {tuple(history.shape)}.'
            )
        if history.shape[1] != self.history_length:
            raise ValueError(
                f'history length must be {self.history_length}, '
                f'got {history.shape[1]}.'
            )
        rps = history[..., :1]
        metrics = history[..., 1:]
        node_input = torch.stack(
            [metrics, rps.expand(-1, -1, self.node_count)], dim=-1
        )
        return node_input, metrics

    # ------------------------------------------------------------------
    # seq2seq
    # ------------------------------------------------------------------
    def encode(self, node_input, graph):
        batch, steps = node_input.shape[:2]
        hidden = self.hidden_init.expand(batch, self.node_count, self.hidden_dim)
        for step in range(steps):
            hidden = self.encoder_cell(node_input[:, step], hidden, graph)
        return hidden

    def decode(self, hidden, last_metric, graph, target=None, steps=None):
        horizon = self.DeltaT if steps is None else int(steps)
        if horizon <= 0 or horizon > self.DeltaT:
            raise ValueError('steps must be in [1, DeltaT].')
        predictions = []
        current = last_metric
        for step in range(horizon):
            if target is not None:
                if target.dim() != 3 or target.shape[1] < horizon:
                    raise ValueError('target must be (batch, DeltaT, nodes) in training.')
                current = target[:, step - 1] if step > 0 else current
            hidden = self.decoder_cell(current.unsqueeze(-1), hidden, graph)
            step = self.output_head(hidden).squeeze(-1)
            current = current + step if self.residual_output else step
            predictions.append(current)
        return torch.stack(predictions, dim=1)

    def forward(self, history, target=None, sample_graph=None, steps=None):
        node_input, metrics = self.prepare_nodes(history)
        graph, _ = self.graph(node_input, sample=sample_graph)
        hidden = self.encode(node_input, graph)
        predictions = self.decode(
            hidden, metrics[:, -1], graph, target=target, steps=steps
        )
        return predictions, graph

    def predict(self, history, steps=None):
        """Autoregressive forecast; ignores future values by construction."""
        predictions, _ = self.forward(
            history, target=None, sample_graph=False, steps=steps
        )
        return predictions

    def graph_from_history(self, history, sample=False):
        node_input, _ = self.prepare_nodes(history)
        return self.graph(node_input, sample=sample)

    # ------------------------------------------------------------------
    # residual-based fault score (unsupervised, validation-calibrated)
    # ------------------------------------------------------------------
    def residual_errors(self, predictions, target):
        if predictions.shape != target.shape:
            raise ValueError('predictions and target must have the same shape.')
        return (predictions - target).abs().mean(dim=1)

    def _aggregate_standardized(self, standardized):
        if self.score_aggregation == 'max':
            return standardized.max(dim=1).values
        if self.score_aggregation == 'mean':
            return standardized.mean(dim=1)
        k = max(1, min(self.score_top_k, standardized.shape[1]))
        return standardized.topk(k, dim=1).values.mean(dim=1)

    def calibrate_from_errors(self, errors):
        """Fit a robust per-node scale and aggregate scale without labels.

        The score of a window is ``1 - exp(-aggregate / scale)``: a window at
        the typical positive calibration anomaly scores about ``0.63``, and
        larger residual deviations score higher.  The map only saturates when
        the residual is many times that scale, so scores stay informative under
        concept drift.
        """
        errors = torch.as_tensor(errors, dtype=torch.float32)
        if errors.dim() != 2 or errors.shape[1] != self.node_count:
            raise ValueError(f'errors must be (windows, {self.node_count}).')
        if errors.shape[0] == 0:
            raise ValueError('Cannot calibrate on an empty residual set.')
        center = errors.median(dim=0).values
        mad = (errors - center).abs().median(dim=0).values
        scale = (1.4826 * mad).clamp_min(self.score_min_scale)
        standardized = F.relu((errors - center) / scale)
        aggregate = self._aggregate_standardized(standardized)
        positive = aggregate[aggregate > 0]
        aggregate_scale = (
            positive.median() if positive.numel() else
            torch.full_like(aggregate[:1], self.score_min_scale)
        ).clamp_min(self.score_min_scale)
        with torch.no_grad():
            self.residual_center.copy_(center.to(self.residual_center.device))
            self.residual_scale.copy_(scale.to(self.residual_scale.device))
            self.residual_aggregate_scale.copy_(
                aggregate_scale.to(self.residual_aggregate_scale.device)
            )
            self.residual_calibrated.fill_(True)

    def standardized_errors(self, predictions, target):
        errors = self.residual_errors(predictions, target)
        scale = self.residual_scale.to(errors.device).clamp_min(self.score_min_scale)
        return (errors - self.residual_center.to(errors.device)) / scale

    def aggregate_from_node_errors(self, node_errors, center=None, scale=None):
        """Robust per-window anomaly magnitude (before the [0, 1] map)."""
        node_errors = torch.as_tensor(node_errors, dtype=self.residual_center.dtype)
        if node_errors.dim() == 1:
            node_errors = node_errors.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        if node_errors.shape[-1] != self.node_count:
            raise ValueError(f'node_errors must have {self.node_count} columns.')
        if center is None:
            center = self.residual_center
        if scale is None:
            scale = self.residual_scale
        center = torch.as_tensor(center, dtype=node_errors.dtype,
                                 device=node_errors.device)
        scale = torch.as_tensor(scale, dtype=node_errors.dtype,
                                device=node_errors.device).clamp_min(
                                    self.score_min_scale)
        standardized = F.relu((node_errors - center) / scale)
        aggregate = self._aggregate_standardized(standardized)
        return aggregate.squeeze(0) if squeeze else aggregate

    def score_from_aggregate(self, aggregate, aggregate_scale=None):
        aggregate = torch.as_tensor(aggregate, dtype=self.residual_aggregate_scale.dtype)
        if aggregate_scale is None:
            if self.residual_calibrated.item():
                aggregate_scale = self.residual_aggregate_scale
            else:
                aggregate_scale = torch.tensor(self.score_temperature)
        aggregate_scale = torch.as_tensor(
            aggregate_scale, dtype=aggregate.dtype, device=aggregate.device,
        ).clamp_min(self.score_min_scale)
        return (1.0 - torch.exp(-aggregate / aggregate_scale)).clamp(0.0, 1.0)

    def score_from_node_errors(self, node_errors, aggregate_scale=None):
        """Map per-window, per-node residuals to a bounded alarm score.

        ``node_errors`` is ``(nodes,)`` or ``(batch, nodes)``.  The score is
        monotone in the residual and stays in ``[0, 1)``; it is causal because
        the trainer only feeds residuals of already observed timestamps.
        """
        aggregate = self.aggregate_from_node_errors(node_errors)
        return self.score_from_aggregate(aggregate, aggregate_scale)

    def residual_scores(self, predictions, target):
        """Convenience wrapper for already observed prediction/target pairs."""
        return self.score_from_node_errors(self.residual_errors(predictions, target))

    def get_calibration(self):
        return {
            'center': self.residual_center.detach().cpu().tolist(),
            'scale': self.residual_scale.detach().cpu().tolist(),
            'aggregate_scale': float(self.residual_aggregate_scale.item()),
            'calibrated': bool(self.residual_calibrated.item()),
        }

    def set_calibration(self, calibration):
        if not calibration:
            return
        with torch.no_grad():
            self.residual_center.copy_(
                torch.as_tensor(calibration['center'], dtype=self.residual_center.dtype)
            )
            self.residual_scale.copy_(
                torch.as_tensor(calibration['scale'], dtype=self.residual_scale.dtype)
            )
            aggregate_scale = calibration.get('aggregate_scale')
            if aggregate_scale is not None:
                self.residual_aggregate_scale.fill_(float(aggregate_scale))
            self.residual_calibrated.fill_(bool(calibration.get('calibrated', True)))

    def method_protocol(self):
        """Checkpoint fingerprint; changing any component invalidates a run."""
        return {
            'method': 'sagn',
            'seq_length': self.seq_length,
            'DeltaT': self.DeltaT,
            'feature_size': self.feature_size,
            'hidden_dim': self.hidden_dim,
            'graph_steps': self.graph_steps,
            'graph_l1': self.graph_l1,
            'gumbel_temperature': self.gumbel_temperature_init,
            'gumbel_temperature_min': self.gumbel_temperature_min,
            'eval_temperature': self.graph.eval_temperature,
            'teacher_forcing': self.teacher_forcing,
            'teacher_forcing_decay_steps': self.teacher_forcing_decay_steps,
            'residual_output': self.residual_output,
            'pool': self.pool_config,
            'online': self.online_config,
            'residual_score': self.residual_config,
        }
