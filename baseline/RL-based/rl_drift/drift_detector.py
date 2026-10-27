"""Fixed-threshold drift detector on natural chronological accuracy.

The paper states that the detector receives the predictor's status but does
not prescribe an algorithm.  The agreed protocol for this repository is a
fixed threshold on rolling accuracy over naturally occurring data; no
synthetic drift is injected.

If a threshold produces fewer than ``min_events`` alerts, the lowest natural
rolling-accuracy minima are used as drift candidates.  Every candidate is an
actual point in the validation drift stream; this is a coverage fallback, not
a synthetic drift.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import Config, DetectorConfig


@dataclass
class DriftEvent:
    event_id: int
    anchor: int                 # index into the drift-stream window array
    available_windows: int      # post-anchor windows available for adaptation
    reason: str
    rolling_accuracy: float
    timestamp: object = None

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "anchor": int(self.anchor),
            "available_windows": int(self.available_windows),
            "reason": self.reason,
            "rolling_accuracy": float(self.rolling_accuracy),
            "timestamp": str(self.timestamp),
        }


@dataclass
class DetectionResult:
    predictions: np.ndarray
    correctness: np.ndarray
    rolling_accuracy: np.ndarray
    alerts: List[int]
    events: List[DriftEvent]
    threshold: float
    window: int


class DriftDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.buffer: deque[float] = deque(maxlen=config.rolling_window)
        self.last_alert: Optional[int] = None
        self.rolling: list[float] = []

    def update(self, correct: bool, index: int) -> Optional[float]:
        """Feed one prediction status; return rolling accuracy or ``None``."""
        self.buffer.append(1.0 if correct else 0.0)
        if len(self.buffer) < self.config.rolling_window:
            return None
        accuracy = float(np.mean(self.buffer))
        self.rolling.append(accuracy)
        if accuracy < self.config.accuracy_threshold:
            if (
                self.last_alert is None
                or index - self.last_alert >= self.config.cooldown
            ):
                self.last_alert = index
                return accuracy
        return None

    def reset(self) -> None:
        self.buffer.clear()
        self.last_alert = None
        self.rolling.clear()


def rolling_accuracy_curve(correctness: np.ndarray, window: int) -> np.ndarray:
    correctness = np.asarray(correctness, dtype=np.float64)
    curve = np.full(correctness.shape, np.nan, dtype=np.float64)
    if correctness.size < window:
        return curve
    cumulative = np.concatenate([[0.0], np.cumsum(correctness)])
    curve[window - 1:] = (cumulative[window:] - cumulative[:-window]) / window
    return curve


def _subsample_alerts(alerts: Sequence[int], max_events: int) -> List[int]:
    alerts = sorted(set(int(a) for a in alerts))
    if len(alerts) <= max_events:
        return alerts
    positions = np.linspace(0, len(alerts) - 1, max_events)
    return [alerts[int(round(p))] for p in positions]


def find_natural_drift_events(
    predictions: np.ndarray,
    labels: np.ndarray,
    config: Config,
    timestamps: Optional[np.ndarray] = None,
) -> DetectionResult:
    """Detect drift candidates using only the validation drift stream."""
    predictions = np.asarray(predictions).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if predictions.size != labels.size:
        raise ValueError("predictions and labels must have the same length.")
    n = labels.size
    if n == 0:
        raise ValueError("The drift stream is empty.")

    detector = DriftDetector(config.detector)
    correctness = (predictions == labels).astype(np.float64)
    rolling = rolling_accuracy_curve(correctness, config.detector.rolling_window)

    raw_alerts: List[int] = []
    for index, ok in enumerate(correctness):
        accuracy = detector.update(bool(ok), index)
        if accuracy is not None:
            raw_alerts.append(index)

    # An offline adaptation always starts from at least D0 samples so that
    # the data collector's action grid exists.  A threshold alert at the very
    # end of the stream cannot supply that much post-drift data.
    min_post = min(
        max(
            int(config.detector.min_post_windows),
            int(config.data_collector.D0),
        ),
        max(1, n - 1),
    )
    alerts = _subsample_alerts(
        [
            idx for idx in raw_alerts
            if idx < n - 1 and (n - idx - 1) >= min_post
        ],
        config.detector.max_events,
    )

    # Coverage fallback: lowest natural rolling-accuracy minima.  They are
    # separated so that adaptation data is not drawn from almost the same
    # instant repeatedly.
    candidates: List[Tuple[int, str, float]] = [
        (idx, "threshold", float(rolling[idx]))
        for idx in alerts
    ]

    if (
        len(candidates) < config.detector.min_events
        and config.detector.fallback_to_accuracy_minima
    ):
        curve = rolling
        valid_indices = np.flatnonzero(
            np.isfinite(curve)
            & (np.arange(n) < n - max(1, min_post))
        )
        minima: List[Tuple[int, float]] = []
        for idx in valid_indices:
            left = curve[idx - 1] if idx - 1 >= 0 else curve[idx]
            right = curve[idx + 1] if idx + 1 < n else curve[idx]
            if curve[idx] <= left and curve[idx] <= right:
                minima.append((int(idx), float(curve[idx])))
        minima.sort(key=lambda item: (item[1], item[0]))
        separation = max(config.detector.rolling_window // 2, 1)
        chosen = [idx for idx, _ in candidates]
        for idx, value in minima:
            if len(chosen) >= config.detector.max_events:
                break
            if all(abs(idx - other) >= separation for other in chosen):
                chosen.append(idx)
                candidates.append((idx, "accuracy_minimum", value))

    if len(candidates) < config.detector.min_events:
        # Final coverage fallback: evenly spaced natural anchors that leave
        # enough post-drift data.  No synthetic samples are generated.
        allowed = np.arange(max(1, n - min_post), dtype=np.int64)
        chosen = [idx for idx, _, _ in candidates]
        if allowed.size:
            count = min(
                config.detector.max_events - len(candidates),
                config.detector.min_events - len(candidates),
            )
            if count > 0:
                picks = np.unique(np.linspace(
                    int(allowed[0]), int(allowed[-1]), count * 3 + len(chosen) + 2
                ).astype(int))
                for idx in picks:
                    idx = int(idx)
                    if len(candidates) >= config.detector.max_events:
                        break
                    if idx in chosen:
                        continue
                    if all(abs(idx - other) >= 1 for other in chosen):
                        value = float(curve[idx]) if np.isfinite(curve[idx]) else float("nan")
                        chosen.append(idx)
                        candidates.append((idx, "even_spacing", value))
                    if len(candidates) >= config.detector.min_events:
                        break

    if not candidates:
        raise ValueError(
            "No natural drift candidate could be formed from the validation "
            "drift stream. Increase the validation segment or lower "
            "detector.min_post_windows."
        )

    candidates.sort(key=lambda item: item[0])
    events: List[DriftEvent] = []
    for event_id, (anchor, reason, accuracy) in enumerate(candidates):
        available = n - (anchor + 1)
        if available <= 0:
            continue
        ts = timestamps[anchor] if timestamps is not None else anchor
        events.append(DriftEvent(
            event_id=event_id,
            anchor=int(anchor),
            available_windows=int(available),
            reason=reason,
            rolling_accuracy=accuracy,
            timestamp=ts,
        ))
    if not events:
        raise ValueError("No drift event has post-drift data available.")

    return DetectionResult(
        predictions=predictions,
        correctness=correctness,
        rolling_accuracy=rolling,
        alerts=alerts,
        events=events,
        threshold=config.detector.accuracy_threshold,
        window=config.detector.rolling_window,
    )
