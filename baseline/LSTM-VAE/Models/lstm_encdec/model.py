"""LSTM Encoder-Decoder for multi-sensor anomaly detection (EncDec-AD).

Implements the reconstruction model of Malhotra et al., "LSTM-based
Encoder-Decoder for Multi-sensor Anomaly Detection" (ICML 2016 workshop).
The encoder maps a window to its final hidden state; the decoder reconstructs
the window in reverse order from that state.  The model is trained only on
normal windows with a reconstruction loss, and anomaly scores are derived from
the reconstruction error by the trainer/report layer.
"""

from __future__ import annotations

import torch
from torch import nn


class EncDecAD(nn.Module):
    """Deterministic LSTM encoder-decoder reconstruction network.

    Parameters
    ----------
    feature_size:
        Number of sensor channels ``m`` in every time step.
    hidden_size:
        Number of LSTM units ``c`` in both the encoder and decoder.
    num_layers:
        Number of stacked LSTM layers for each network.
    dropout:
        Dropout between stacked LSTM layers (ignored for a single layer).
    """

    def __init__(self, feature_size, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        if not isinstance(feature_size, int) or feature_size <= 0:
            raise ValueError('feature_size must be a positive integer.')
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError('hidden_size must be a positive integer.')
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError('num_layers must be a positive integer.')
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError('dropout must be in [0, 1).')

        self.feature_size = int(feature_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        recurrent_dropout = self.dropout if self.num_layers > 1 else 0.0

        self.encoder = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            input_size=self.feature_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_layer = nn.Linear(self.hidden_size, self.feature_size)

    def forward(self, x, teacher_forcing=True):
        """Reconstruct ``x`` in reverse time order.

        ``x`` has shape ``[batch, length, feature_size]``.  The encoder's final
        hidden state predicts the last point.  For ``i = length - 2 ... 0`` the
        decoder consumes either the observed ``x[i + 1]`` (teacher forcing,
        training) or the prediction ``x'[i + 1]`` (inference) and predicts
        ``x'[i]``.  The returned tensor preserves the original time order.
        """
        if x.dim() != 3:
            raise ValueError('Expected a [batch, length, feature_size] tensor.')
        batch, length, features = x.shape
        if features != self.feature_size:
            raise ValueError(
                f'Expected {self.feature_size} features per time step, got {features}.'
            )
        if length < 1:
            raise ValueError('The input window must contain at least one time step.')

        _, (hidden, cell) = self.encoder(x)
        reconstruction = x.new_empty(batch, length, features)
        reconstruction[:, -1] = self.output_layer(hidden[-1])

        state = (hidden, cell)
        previous = x[:, -1] if teacher_forcing else reconstruction[:, -1].detach()
        for step in range(length - 2, -1, -1):
            output, state = self.decoder(previous.unsqueeze(1), state)
            reconstruction[:, step] = self.output_layer(output[:, 0])
            previous = (
                x[:, step] if teacher_forcing else reconstruction[:, step].detach()
            )
        return reconstruction

    @torch.no_grad()
    def reconstruct(self, x):
        """Inference-time reconstruction using the model's own predictions."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(x, teacher_forcing=False)
        finally:
            self.train(was_training)
