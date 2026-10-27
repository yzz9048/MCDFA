"""CNN fault predictor and P/D model bank.

Architecture follows the paper's selected predictor:

    Conv1d -> ReLU -> Conv1d -> ReLU -> MaxPool1d -> Flatten
           -> Dense -> ReLU -> Dropout -> Dense(2)

The four trainable layer groups (``conv1``, ``conv2``, ``dense1``,
``dense2``) are the coarse-grained modules used by the R/F/Z actions.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional

import torch
from torch import nn

from .config import PredictorConfig

LAYER_NAMES: tuple[str, ...] = ("conv1", "conv2", "dense1", "dense2")
LAYER_ACTIONS: tuple[str, ...] = ("R", "F", "Z")
TERMINAL_ACTIONS: tuple[str, ...] = ("P", "D")


def _conv_output_length(input_window: int, kernel_size: int) -> int:
    length = input_window
    for _ in range(2):
        length = length - (kernel_size - 1)
    return length


class FaultCNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        input_window: int = 12,
        config: Optional[PredictorConfig] = None,
        num_classes: int = 2,
    ):
        super().__init__()
        cfg = config or PredictorConfig()
        conv_len = _conv_output_length(input_window, cfg.kernel_size)
        pooled_len = conv_len // cfg.pool_size
        if pooled_len <= 0:
            raise ValueError(
                f"input_window={input_window} is too short for two "
                f"kernel_size={cfg.kernel_size} convolutions and pool={cfg.pool_size}."
            )
        self.input_dim = int(input_dim)
        self.input_window = int(input_window)
        self.config = cfg
        self.conv1 = nn.Conv1d(input_dim, cfg.conv1_out, cfg.kernel_size)
        self.conv2 = nn.Conv1d(cfg.conv1_out, cfg.conv2_out, cfg.kernel_size)
        self.pool = nn.MaxPool1d(cfg.pool_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc_in = cfg.conv2_out * pooled_len
        self.dense1 = nn.Linear(self.fc_in, cfg.dense1)
        self.dense2 = nn.Linear(cfg.dense1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, F) -> (B, F, T)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = torch.relu(self.dense1(x))
        x = self.dropout(x)
        return self.dense2(x)

    # -------------------------------------------------------------- layers
    def layer_modules(self, name: str) -> List[nn.Module]:
        mapping = {
            "conv1": [self.conv1],
            "conv2": [self.conv2],
            "dense1": [self.dense1],
            "dense2": [self.dense2],
        }
        if name not in mapping:
            raise KeyError(f"Unknown layer group {name!r}.")
        return mapping[name]

    def layer_parameters(self, name: str) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for module in self.layer_modules(name):
            params.extend(list(module.parameters()))
        return params

    def reset_layer(self, name: str) -> None:
        for module in self.layer_modules(name):
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            else:  # pragma: no cover - all current groups expose reset
                for param in module.parameters():
                    nn.init.zeros_(param)

    def reset_all_layers(self) -> None:
        for name in LAYER_NAMES:
            self.reset_layer(name)

    def set_action_mask(self, actions: Dict[str, str]) -> None:
        """Apply F/Z action semantics to parameter gradients.

        ``F`` keeps old weights and trains the group, ``Z`` freezes it.  A
        complete ``R`` plan resets every group before this method is called.
        """
        for name, action in actions.items():
            if action not in LAYER_ACTIONS:
                raise ValueError(f"Unknown layer action {action!r} for {name!r}.")
            requires_grad = action in {"R", "F"}
            for param in self.layer_parameters(name):
                param.requires_grad = requires_grad

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return (p for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        self.eval()
        chunks = []
        for start in range(0, x.shape[0], batch_size):
            logits = self(x[start:start + batch_size])
            chunks.append(torch.softmax(logits, dim=1))
        if not chunks:
            return torch.empty((0, 2), device=x.device)
        return torch.cat(chunks, dim=0)

    def clone(self) -> "FaultCNN":
        other = FaultCNN(
            self.input_dim,
            self.input_window,
            config=self.config,
            num_classes=self.dense2.out_features,
        )
        other.load_state_dict(copy.deepcopy(self.state_dict()))
        other.to(next(self.parameters()).device)
        return other

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ModelBank:
    """Current model plus preserved old models for the ``P`` action.

    ``D`` replaces the bank with the newly adapted model and clears preserved
    models.  ``P`` keeps the old current model in the preserved set.
    Inference averages class probabilities over all bank members.
    """

    def __init__(self, initial: FaultCNN):
        self.current = initial
        self.preserved: List[FaultCNN] = []

    def members(self) -> List[FaultCNN]:
        return [self.current, *self.preserved]

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
        if not self.preserved:
            return self.current.predict_proba(x, batch_size=batch_size)
        probs = [model.predict_proba(x, batch_size=batch_size) for model in self.members()]
        return torch.stack(probs, dim=0).mean(dim=0)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.predict_proba(x).argmax(dim=1)

    def replace(self, new_model: FaultCNN, preserve_old: bool) -> None:
        if preserve_old:
            self.preserved.append(self.current)
        else:
            self.preserved.clear()
        self.current = new_model

    def clone(self) -> "ModelBank":
        bank = ModelBank(self.current.clone())
        bank.preserved = [model.clone() for model in self.preserved]
        return bank

    def to(self, device: torch.device | str) -> "ModelBank":
        self.current.to(device)
        for model in self.preserved:
            model.to(device)
        return self

    def state(self) -> Dict[str, object]:
        return {
            "current": copy.deepcopy(self.current.state_dict()),
            "preserved": [copy.deepcopy(m.state_dict()) for m in self.preserved],
            "input_dim": self.current.input_dim,
            "input_window": self.current.input_window,
            "config": self.current.config,
        }

    def load_state(self, state: Dict[str, object]) -> None:
        self.current = FaultCNN(
            int(state["input_dim"]), int(state["input_window"]),
            config=state["config"],  # type: ignore[arg-type]
        )
        self.current.load_state_dict(state["current"])  # type: ignore[arg-type]
        self.preserved = []
        for weights in state["preserved"]:  # type: ignore[union-attr]
            model = FaultCNN(
                int(state["input_dim"]), int(state["input_window"]),
                config=state["config"],  # type: ignore[arg-type]
            )
            model.load_state_dict(weights)
            self.preserved.append(model)
        device = torch.device("cpu")
        self.to(device)


def accuracy_from_proba(proba: torch.Tensor, labels: torch.Tensor) -> float:
    if labels.numel() == 0:
        return float("nan")
    return float((proba.argmax(dim=1) == labels).float().mean().item())
