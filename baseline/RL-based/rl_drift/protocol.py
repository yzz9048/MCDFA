"""Algorithm 1 online simulation and frozen-test evaluation.

The online handler only receives the validation drift stream.  Test data is
passed exclusively to ``evaluate_frozen_model`` after adaptation is finished;
that function performs inference, never parameter updates.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .adaptation import AdaptationEnvironment, classification_metrics
from .config import Config, KPISet
from .data import PaperDataModule
from .drift_detector import DriftDetector, DriftEvent
from .model import FaultCNN, ModelBank
from .q_learning import DataCollectorAgent, MethodSelectorAgent
from .utils import json_safe, write_json


@dataclass
class OnlineResult:
    bank: ModelBank
    records: List[Dict[str, object]]
    adaptations: List[Dict[str, object]]
    initial_holdout: Dict[str, float]
    final_holdout: Dict[str, float]
    stream_metrics: Dict[str, float]
    consumed_windows: int
    alerts: int

    def summary(self) -> Dict[str, object]:
        return json_safe({
            "initial_holdout": self.initial_holdout,
            "final_holdout": self.final_holdout,
            "stream_metrics": self.stream_metrics,
            "consumed_windows": self.consumed_windows,
            "alerts": self.alerts,
            "adaptations": self.adaptations,
        })


class OnlineDriftHandler:
    def __init__(
        self,
        config: Config,
        data: PaperDataModule,
        initial_model: FaultCNN,
        selector: MethodSelectorAgent,
        collector_lookup: Callable[[object, int], DataCollectorAgent],
        environment: AdaptationEnvironment,
        device: torch.device,
        kpi: Optional[KPISet] = None,
    ):
        self.config = config
        self.data = data
        self.device = device
        self.selector = selector
        self.collector_lookup = collector_lookup
        self.environment = environment
        self.kpi = kpi or config.kpi()
        self.kpi_index = config.kpi_sets.index(self.kpi)
        self.initial_model = initial_model.clone().to(device)
        self.initial_model.eval()

    def _choose_data_size(
        self,
        collector: DataCollectorAgent,
        grid: Sequence[int],
        event: DriftEvent,
    ) -> int:
        if not grid:
            raise ValueError("Empty data grid.")
        if len(grid) == 1:
            return int(grid[0])
        target = int(self.config.data_collector.deployment_initial_size)
        size = min(grid, key=lambda value: abs(int(value) - target))
        seen = {int(size)}
        for _ in range(int(self.config.qlearning.collector_episode_length)):
            actions = collector.available_actions(size, grid)
            if not actions:
                break
            values = [collector.q_value(size, action) for action in actions]
            if max(values) == 0.0 and min(values) == 0.0:
                break  # untrained / unvisited: fall back to configured size
            action = collector.select_action(size, grid, epsilon=0.0, greedy=True)
            if action is None:
                break
            next_size = collector.next_state(size, action, grid)
            if next_size is None or next_size in seen:
                break
            seen.add(int(next_size))
            size = int(next_size)
        return int(size)

    def run(self) -> OnlineResult:
        stream = self.data.stream_windows
        x_all = torch.from_numpy(stream.features).to(self.device)
        y_all = torch.from_numpy(stream.labels).long().to(self.device)
        n = int(x_all.shape[0])
        bank = ModelBank(self.initial_model.clone())
        bank.to(self.device)
        detector = DriftDetector(self.config.detector)
        records: List[Dict[str, object]] = []
        adaptations: List[Dict[str, object]] = []
        correctness: List[float] = []
        consumed = 0
        skipped_alerts = 0
        index = 0
        while index < n:
            with torch.no_grad():
                probabilities = bank.predict_proba(x_all[index:index + 1])
                prediction = int(probabilities.argmax(dim=1).item())
            label = int(y_all[index].item())
            correct = prediction == label
            alert_accuracy = detector.update(correct, index)
            rolling = (
                float(np.mean(detector.buffer))
                if len(detector.buffer) >= self.config.detector.rolling_window
                else float("nan")
            )
            correctness.append(1.0 if correct else 0.0)
            records.append({
                "stream_index": index,
                "timestamp": str(stream.timestamps[index]),
                "start": int(stream.starts[index]),
                "prediction": prediction,
                "label": label,
                "correct": int(correct),
                "rolling_accuracy": rolling,
                "alert": int(alert_accuracy is not None),
            })

            if alert_accuracy is not None and index < n - 1:
                plan = self.selector.greedy_plan(self.kpi_index)
                collector = self.collector_lookup(plan, self.kpi_index)
                event = DriftEvent(
                    event_id=1_000_000 + index,
                    anchor=index,
                    available_windows=n - index - 1,
                    reason="online_threshold",
                    rolling_accuracy=float(alert_accuracy),
                    timestamp=stream.timestamps[index],
                )
                grid = self.environment.data_grid(event)
                min_data = int(self.config.data_collector.D0)
                if event.available_windows < min_data:
                    # D0 is the paper's minimum adaptation-data size.  Near
                    # the end of the validation stream there is no valid
                    # data-collector action; skip rather than train on a
                    # handful of windows.
                    skipped_alerts += 1
                    index += 1
                    continue
                if grid:
                    data_size = self._choose_data_size(collector, grid, event)
                    before = self.environment.evaluate_bank_on_holdout(bank)
                    bank = self.environment.adapt_bank(
                        bank, plan, event, data_size,
                        seed_tag=("online", index, plan.name, data_size),
                    )
                    after = self.environment.evaluate_bank_on_holdout(bank)
                    adaptations.append({
                        "stream_index": index,
                        "timestamp": str(stream.timestamps[index]),
                        "rolling_accuracy": float(alert_accuracy),
                        "plan": plan.name,
                        "data_size": int(data_size),
                        "available_windows": int(event.available_windows),
                        "holdout_accuracy_before": before["accuracy"],
                        "holdout_accuracy_after": after["accuracy"],
                        "bank_members": len(bank.members()),
                    })
                    detector.reset()
                    # Windows ``index + 1 ... index + data_size`` were used for
                    # adaptation; the next online observation is after them.
                    consumed += data_size
                    index = min(n, index + data_size + 1)
                    continue
            index += 1

        stream_accuracy = float(np.mean(correctness)) if correctness else float("nan")
        initial_holdout = self.environment.initial_holdout_metrics()
        final_holdout = self.environment.evaluate_bank_on_holdout(bank)
        stream_metrics = {
            "evaluated_windows": len(correctness),
            "skipped_for_adaptation": consumed,
            "skipped_alerts": skipped_alerts,
            "accuracy": stream_accuracy,
            "alerts": len(adaptations),
        }
        return OnlineResult(
            bank=bank,
            records=records,
            adaptations=adaptations,
            initial_holdout=initial_holdout,
            final_holdout=final_holdout,
            stream_metrics=stream_metrics,
            consumed_windows=consumed,
            alerts=len(adaptations),
        )


def _as_bank(model_or_bank: FaultCNN | ModelBank) -> ModelBank:
    if isinstance(model_or_bank, ModelBank):
        return model_or_bank
    return ModelBank(model_or_bank)


@torch.no_grad()
def evaluate_frozen_model(
    config: Config,
    data: PaperDataModule,
    model_or_bank: FaultCNN | ModelBank,
    device: torch.device,
) -> Dict[str, object]:
    """Inference-only test evaluation.  No gradients or updates occur."""
    bank = _as_bank(model_or_bank)
    x = torch.from_numpy(data.test_windows.features).to(device)
    y = torch.from_numpy(data.test_windows.labels).long().to(device)
    probabilities = bank.predict_proba(x)
    metrics = classification_metrics(probabilities, y)
    predictions = probabilities.argmax(dim=1).cpu().numpy().astype(np.int64)

    result: Dict[str, object] = dict(metrics)
    result["windows"] = int(y.numel())
    result["positives"] = int(y.sum().item())

    # Event-level metrics use the repository's existing one-to-one matcher.
    # This is metric computation only; it never feeds adaptation.
    try:
        from Utils.lead_time import (
            compute_event_classification_metrics, compute_event_lead_time,
        )
        total_window = config.predictor.input_window + config.predictor.forecast_horizon
        lead_average, detected, eligible = compute_event_lead_time(
            predictions,
            data.test_windows.starts,
            data.raw_labels,
            total_window,
            config.predictor.forecast_horizon,
        )
        result.update({
            "event_avg_lead_time_timestamps": lead_average,
            "detected_onsets": detected,
            "eligible_onsets": eligible,
        })
        result.update(compute_event_classification_metrics(
            predictions,
            data.test_windows.starts,
            data.raw_labels,
            total_window,
            config.predictor.forecast_horizon,
        ))
    except Exception as exc:  # pragma: no cover - optional legacy utility
        result["event_metrics_error"] = str(exc)
    return result


def save_online_result(
    path_dir: Path,
    result: OnlineResult,
) -> None:
    path_dir.mkdir(parents=True, exist_ok=True)
    if result.records:
        with open(path_dir / "online_stream.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.records[0]))
            writer.writeheader()
            writer.writerows(result.records)
    if result.adaptations:
        with open(path_dir / "online_adaptations.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.adaptations[0]))
            writer.writeheader()
            writer.writerows(result.adaptations)
    write_json(path_dir / "online_summary.json", result.summary())
