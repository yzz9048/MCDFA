"""Adaptation plans, model training, KPI measurement, and reward.

Layer actions (paper Eq. 5):
    R  retrain: reset the group and train it
    F  fine-tune: keep old weights, train the group
    Z  freeze: keep old weights, no gradients
Terminal actions:
    P  preserve old model in an ensemble with the new model
    D  discard old model, keep only the new model

The environment is trained from the initial predictor for offline RL.  The
online handler adapts the current bank and uses the same training routine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .config import Config, KPISet
from .data import PaperDataModule
from .drift_detector import DriftEvent
from .model import (
    LAYER_ACTIONS, LAYER_NAMES, TERMINAL_ACTIONS, FaultCNN, ModelBank,
)
from .utils import (
    ResourceMeter, Stopwatch, json_safe, stable_seed, temporary_seed,
    write_json,
)


@dataclass(frozen=True)
class AdaptationPlan:
    layers: Tuple[str, ...]
    terminal: str

    def __post_init__(self) -> None:
        if len(self.layers) != len(LAYER_NAMES):
            raise ValueError(
                f"A plan needs {len(LAYER_NAMES)} layer actions, got {self.layers}."
            )
        if self.terminal not in TERMINAL_ACTIONS:
            raise ValueError(f"Unknown terminal action {self.terminal!r}.")
        if any(action not in LAYER_ACTIONS for action in self.layers):
            raise ValueError(f"Unknown layer action in {self.layers}.")
        if self.layers[0] == "R" and any(action != "R" for action in self.layers):
            raise ValueError("After an initial R, every layer action must be R.")
        if self.layers[0] != "R" and "R" in self.layers:
            raise ValueError("R is only available as the first layer action.")

    @property
    def name(self) -> str:
        return ",".join(self.layers) + "|" + self.terminal

    @property
    def action_map(self) -> Dict[str, str]:
        return dict(zip(LAYER_NAMES, self.layers))

    @classmethod
    def from_name(cls, name: str) -> "AdaptationPlan":
        head, terminal = name.split("|")
        return cls(tuple(head.split(",")), terminal)

    @classmethod
    def all_plans(cls) -> List["AdaptationPlan"]:
        plans: List[AdaptationPlan] = []
        # Initial R forces R for all remaining layers.
        for terminal in TERMINAL_ACTIONS:
            plans.append(cls(("R", "R", "R", "R"), terminal))
        # Initial F or Z, then F/Z for every remaining layer.
        for first in ("F", "Z"):
            for tail in np.ndindex(*(2,) * (len(LAYER_NAMES) - 1)):
                layers = (first,) + tuple(
                    "F" if index == 0 else "Z" for index in tail
                )
                for terminal in TERMINAL_ACTIONS:
                    plans.append(cls(layers, terminal))
        return plans


@dataclass
class AdaptationMetrics:
    event_id: int
    plan: str
    data_size: int
    requested_data_size: int
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    resource_mb: float
    footprint_mb: float
    rss_mb: float
    cuda_mb: float
    loss: float
    train_windows: int
    epochs: int
    cached: bool = False
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return json_safe(self.__dict__)


def classification_metrics(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, float]:
    labels = labels.reshape(-1).long()
    if labels.numel() == 0:
        return {
            "accuracy": float("nan"), "balanced_accuracy": float("nan"),
            "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
        }
    predictions = probabilities.argmax(dim=1)
    tp = int(((predictions == 1) & (labels == 1)).sum().item())
    tn = int(((predictions == 0) & (labels == 0)).sum().item())
    fp = int(((predictions == 1) & (labels == 0)).sum().item())
    fn = int(((predictions == 0) & (labels == 1)).sum().item())
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total
    positive_recall = tp / (tp + fn) if tp + fn else float("nan")
    negative_recall = tn / (tn + fp) if tn + fp else float("nan")
    if np.isfinite(positive_recall) and np.isfinite(negative_recall):
        balanced = 0.5 * (positive_recall + negative_recall)
    elif np.isfinite(positive_recall):
        balanced = positive_recall
    elif np.isfinite(negative_recall):
        balanced = negative_recall
    else:
        balanced = float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = positive_recall if np.isfinite(positive_recall) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(labels.long(), minlength=num_classes).float()
    total = counts.sum()
    weights = torch.zeros(num_classes, device=labels.device)
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    return weights


@torch.no_grad()
def _mean_proba(
    models: Sequence[FaultCNN],
    features: torch.Tensor,
    batch_size: int = 512,
) -> torch.Tensor:
    probs = [model.predict_proba(features, batch_size=batch_size) for model in models]
    return torch.stack(probs, dim=0).mean(dim=0)


class AdaptationEnvironment:
    def __init__(
        self,
        config: Config,
        data: PaperDataModule,
        events: Sequence[DriftEvent],
        initial_model: FaultCNN,
        device: torch.device,
    ):
        self.config = config
        self.data = data
        self.events = list(events)
        self.event_by_id = {event.event_id: event for event in self.events}
        self.device = device
        self.base_model = initial_model.clone().to(device)
        self.base_model.eval()

        self.stream_x = torch.from_numpy(data.stream_windows.features).to(device)
        self.stream_y = torch.from_numpy(data.stream_windows.labels).long().to(device)
        self.holdout_x = torch.from_numpy(data.accuracy_windows.features).to(device)
        self.holdout_y = torch.from_numpy(data.accuracy_windows.labels).long().to(device)
        self.cache: Dict[Tuple[int, str, int], AdaptationMetrics] = {}
        self._initial_holdout: Optional[Dict[str, float]] = None

        if self.stream_x.shape[0] < 2:
            raise ValueError("The drift stream needs at least two windows.")

    # -------------------------------------------------------------- data
    def post_data(
        self,
        event: DriftEvent,
        requested_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        start = event.anchor + 1
        available = int(self.stream_x.shape[0]) - start
        if available <= 0:
            raise ValueError(f"Event {event.event_id} has no post-drift data.")
        actual = int(min(max(1, int(requested_size)), available))
        return (
            self.stream_x[start:start + actual],
            self.stream_y[start:start + actual],
            actual,
        )

    def data_grid(self, event: DriftEvent) -> List[int]:
        available = min(
            int(event.available_windows),
            int(self.config.adaptation.max_data_size),
            int(self.config.data_collector.Dr),
        )
        if available <= 0:
            return []
        interval = int(self.config.data_collector.interval)
        start = min(int(self.config.data_collector.D0), available)
        values = list(range(start, available + 1, interval))
        if not values:
            values = [available]
        if values[-1] > available:
            values = [value for value in values if value <= available] or [available]
        return values

    def selector_data_size(self, event: DriftEvent) -> int:
        grid = self.data_grid(event)
        if not grid:
            raise ValueError(f"Event {event.event_id} has no adaptation data.")
        requested = self.config.qlearning.selector_data_size
        if requested == "max":
            return int(grid[-1])
        return int(np.clip(int(requested), grid[0], grid[-1]))

    # ------------------------------------------------------------- model
    def _new_model(self, plan: AdaptationPlan) -> FaultCNN:
        model = self.base_model.clone().to(self.device)
        if plan.layers[0] == "R":
            model.reset_all_layers()
        model.set_action_mask(plan.action_map)
        return model

    def _train(
        self,
        model: FaultCNN,
        features: torch.Tensor,
        labels: torch.Tensor,
        seed: int,
    ) -> float:
        parameters = list(model.trainable_parameters())
        if not parameters:
            model.eval()
            return 0.0
        optimizer = torch.optim.Adam(
            parameters,
            lr=self.config.adaptation.lr,
            weight_decay=self.config.adaptation.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss(weight=_class_weights(labels))
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        n = int(labels.shape[0])
        batch_size = max(1, int(self.config.adaptation.batch_size))
        losses: List[float] = []
        model.train()
        for _ in range(int(self.config.adaptation.epochs)):
            order = torch.randperm(n, generator=generator)
            for batch_start in range(0, n, batch_size):
                index = order[batch_start:batch_start + batch_size].to(self.device)
                logits = model(features[index])
                loss = loss_fn(logits, labels[index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().item()))
        model.eval()
        return float(np.mean(losses)) if losses else 0.0

    def _terminal_probabilities(
        self,
        adapted: FaultCNN,
        terminal: str,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if terminal == "P":
            return _mean_proba([self.base_model, adapted], features)
        return adapted.predict_proba(features)

    def _memory_footprint_mb(
        self,
        model: FaultCNN,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """Deterministic adaptation memory: data + parameters + one batch."""
        parameter_bytes = sum(
            param.numel() * param.element_size() for param in model.parameters()
        )
        data_bytes = (
            features.numel() * features.element_size()
            + labels.numel() * labels.element_size()
        )
        cfg = model.config
        conv_len = model.input_window - 2 * (cfg.kernel_size - 1)
        pooled_len = max(1, conv_len // cfg.pool_size)
        batch = min(max(1, int(self.config.adaptation.batch_size)), int(labels.shape[0]))
        activation_elements = batch * (
            cfg.conv1_out * conv_len
            + cfg.conv2_out * pooled_len
            + model.fc_in
            + cfg.dense1
        )
        activation_bytes = activation_elements * 4
        return float(
            (parameter_bytes + data_bytes + activation_bytes) / (1024.0 ** 2)
        )

    # ---------------------------------------------------------- evaluation
    def initial_holdout_metrics(self) -> Dict[str, float]:
        if self._initial_holdout is None:
            probabilities = self.base_model.predict_proba(self.holdout_x)
            self._initial_holdout = classification_metrics(
                probabilities, self.holdout_y
            )
        return dict(self._initial_holdout)

    def evaluate(
        self,
        plan: AdaptationPlan,
        event: DriftEvent,
        data_size: int,
    ) -> AdaptationMetrics:
        requested = int(data_size)
        key = (int(event.event_id), plan.name, requested)
        if self.config.adaptation.cache and key in self.cache:
            cached = replace(self.cache[key], cached=True)
            return cached

        features, labels, actual = self.post_data(event, requested)
        seed = stable_seed(
            "adaptation", self.config.seed, event.event_id, plan.name, requested
        )
        loss = 0.0
        footprint = 0.0
        with temporary_seed(seed):
            with Stopwatch() as watch, ResourceMeter(self.device) as meter:
                model = self._new_model(plan)
                footprint = self._memory_footprint_mb(model, features, labels)
                loss = self._train(model, features, labels, seed=seed)
                probabilities = self._terminal_probabilities(
                    model, plan.terminal, self.holdout_x
                )
                scores = classification_metrics(probabilities, self.holdout_y)

        resource = (
            footprint
            if self.config.adaptation.resource_metric == "footprint"
            else float(meter.resource_mb)
        )
        metrics = AdaptationMetrics(
            event_id=int(event.event_id),
            plan=plan.name,
            data_size=int(actual),
            requested_data_size=requested,
            accuracy=scores["accuracy"],
            balanced_accuracy=scores["balanced_accuracy"],
            precision=scores["precision"],
            recall=scores["recall"],
            f1=scores["f1"],
            train_seconds=float(watch.elapsed),
            resource_mb=float(resource),
            footprint_mb=float(footprint),
            rss_mb=float(meter.rss_delta_mb),
            cuda_mb=float(meter.cuda_delta_mb),
            loss=float(loss),
            train_windows=actual,
            epochs=int(self.config.adaptation.epochs),
            cached=False,
        )
        if self.config.adaptation.cache:
            self.cache[key] = metrics
        return metrics

    def reward(self, metrics: AdaptationMetrics, kpi: KPISet) -> float:
        """Paper Eq. (6): weighted accuracy gain minus time/resource cost."""
        accuracy = float(metrics.accuracy)
        if not np.isfinite(accuracy):
            return float("-inf")
        required = max(float(kpi.accuracy), 1e-9)
        time_ratio = float(metrics.train_seconds) / max(float(kpi.time), 1e-9)
        resource_ratio = float(metrics.resource_mb) / max(float(kpi.resource), 1e-9)
        cfg = self.config.reward
        return float(
            cfg.k1 * (accuracy - required) / required
            - cfg.k2 * time_ratio
            - cfg.k3 * resource_ratio
        )

    def evaluate_and_reward(
        self,
        plan: AdaptationPlan,
        event: DriftEvent,
        data_size: int,
        kpi: KPISet,
    ) -> Tuple[AdaptationMetrics, float]:
        metrics = self.evaluate(plan, event, data_size)
        return metrics, self.reward(metrics, kpi)

    def accuracy_only(
        self,
        plan: AdaptationPlan,
        event: DriftEvent,
        data_size: int,
    ) -> float:
        return self.evaluate(plan, event, data_size).accuracy

    # --------------------------------------------------------- online use
    def adapt_bank(
        self,
        bank: ModelBank,
        plan: AdaptationPlan,
        event: DriftEvent,
        data_size: int,
        seed_tag: object = "online",
    ) -> ModelBank:
        features, labels, _ = self.post_data(event, data_size)
        seed = stable_seed("online", self.config.seed, seed_tag, plan.name, data_size)
        with temporary_seed(seed):
            model = bank.current.clone().to(self.device)
            if plan.layers[0] == "R":
                model.reset_all_layers()
            model.set_action_mask(plan.action_map)
            self._train(model, features, labels, seed=seed)
        bank.replace(model, preserve_old=(plan.terminal == "P"))
        return bank

    def evaluate_bank_on_holdout(self, bank: ModelBank) -> Dict[str, float]:
        probabilities = bank.predict_proba(self.holdout_x)
        return classification_metrics(probabilities, self.holdout_y)

    # -------------------------------------------------------------- cache
    def cache_rows(self) -> List[Dict[str, object]]:
        return [metrics.as_dict() for metrics in self.cache.values()]

    def save_cache(self, path: Path) -> None:
        write_json(path, self.cache_rows())

    def load_cache(self, path: Path) -> int:
        """Load a cache written by ``save_cache``.

        Only entries for the events in this environment are retained.  This
        lets ``--stage evaluate`` reuse the exact adaptation outcomes from
        ``--stage train-rl`` without recomputing them.
        """
        if not path.exists():
            return 0
        try:
            with open(path, encoding="utf-8") as handle:
                rows = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return 0
        if not isinstance(rows, list):
            return 0
        loaded = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            event_id = row.get("event_id")
            if event_id is None or int(event_id) not in self.event_by_id:
                continue
            plan = row.get("plan")
            requested = row.get("requested_data_size")
            if plan is None or requested is None:
                continue
            values = dict(row)
            for name in (
                "accuracy", "balanced_accuracy", "precision", "recall", "f1",
                "train_seconds", "resource_mb", "footprint_mb", "rss_mb",
                "cuda_mb", "loss",
            ):
                value = values.get(name)
                if value is None:
                    values[name] = float("nan")
                else:
                    values[name] = float(value)
            if not math.isfinite(float(values["accuracy"])):
                continue
            try:
                metrics = AdaptationMetrics(
                    event_id=int(event_id),
                    plan=str(plan),
                    data_size=int(row["data_size"]),
                    requested_data_size=int(requested),
                    accuracy=float(values["accuracy"]),
                    balanced_accuracy=float(values["balanced_accuracy"]),
                    precision=float(values["precision"]),
                    recall=float(values["recall"]),
                    f1=float(values["f1"]),
                    train_seconds=float(values["train_seconds"]),
                    resource_mb=float(values["resource_mb"]),
                    footprint_mb=float(values["footprint_mb"]),
                    rss_mb=float(values["rss_mb"]),
                    cuda_mb=float(values["cuda_mb"]),
                    loss=float(values["loss"]),
                    train_windows=int(row.get("train_windows", row["data_size"])),
                    epochs=int(row.get("epochs", self.config.adaptation.epochs)),
                    cached=bool(row.get("cached", False)),
                    extra=dict(row.get("extra") or {}),
                )
            except (KeyError, TypeError, ValueError):
                continue
            self.cache[(metrics.event_id, metrics.plan, metrics.requested_data_size)] = metrics
            loaded += 1
        return loaded
