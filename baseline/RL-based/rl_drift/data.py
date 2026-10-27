"""Chronological data pipeline for the paper's CNN predictor.

Protocol
--------
Raw timeline (no shuffling anywhere)::

    |<--- 60% train --->|<--- 20% validation --->|<--- 20% test --->|
       |<- fit ->|<-ES->|  <- drift stream ->|<-acc->|
                         (used by RL)          (rewards)

* A window is ``input_window`` timestamps (paper: 12).
* Its label is "at least one fault in the next ``forecast_horizon``
  timestamps" (paper: 6).  This is pure fault classification; no metric
  forecasting.
* RFE and standardisation are fitted on the fit slice only.
* Windows never cross a segment boundary.
* Test windows are constructed but never passed to the detector, adaptation
  environment, reward, or Q-learning code.  They are only read by the final
  frozen evaluation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import Config


@dataclass
class WindowSet:
    features: np.ndarray  # float32 (N, input_window, n_selected_features)
    labels: np.ndarray    # int64 (N,) 1 = fault inside forecast horizon
    starts: np.ndarray    # int64 (N,) raw-timeline start index
    timestamps: np.ndarray  # object (N,) timestamp of window start

    def __len__(self) -> int:
        return int(self.labels.size)

    def empty(self) -> bool:
        return self.labels.size == 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_windows(
    features: np.ndarray,
    labels: np.ndarray,
    segment_start: int,
    segment_end: int,
    input_window: int,
    forecast_horizon: int,
    timestamps: Optional[np.ndarray] = None,
) -> WindowSet:
    """Build non-overlapping-boundary windows fully contained in a segment."""
    _require(segment_end <= features.shape[0], "Segment end exceeds data length.")
    _require(segment_start >= 0 and segment_end > segment_start, "Invalid segment.")
    last_start = segment_end - input_window - forecast_horizon
    if last_start < segment_start:
        return WindowSet(
            features=np.empty((0, input_window, features.shape[1]), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
            starts=np.empty((0,), dtype=np.int64),
            timestamps=np.empty((0,), dtype=object),
        )
    starts = np.arange(segment_start, last_start + 1, dtype=np.int64)
    # Advanced indexing copies into one contiguous array once.
    offsets = np.arange(input_window, dtype=np.int64)
    x = features[starts[:, None] + offsets[None, :]]
    target_offsets = np.arange(input_window, input_window + forecast_horizon)
    future = labels[starts[:, None] + target_offsets[None, :]]
    y = (future != 0).any(axis=1).astype(np.int64)
    ts = (
        timestamps[starts] if timestamps is not None
        else np.asarray(starts, dtype=object)
    )
    return WindowSet(
        features=np.ascontiguousarray(x, dtype=np.float32),
        labels=y,
        starts=starts,
        timestamps=ts,
    )


class PaperDataModule:
    """Load one CSV and expose the paper's chronological split.

    ``test`` is stored separately.  Callers must not pass it to any adaptation
    or RL-training object; ``assert_no_test_leakage`` checks index-level
    disjointness.
    """

    def __init__(self, config: Config):
        self.config = config
        self.data_root = Path(config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.data_root}")

        self.raw_frame = pd.read_csv(self.data_root)
        if self.raw_frame.shape[1] < 3:
            raise ValueError("Dataset must contain timestamp, features, and label.")
        self.timestamp_values = self.raw_frame.iloc[:, 0].astype(str).to_numpy()
        values = self.raw_frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        if values.isna().any().any():
            raise ValueError("Dataset contains non-numeric or missing values.")
        self.raw = values.to_numpy(dtype=np.float64)
        self.feature_names = list(values.columns[:-1])
        self.raw_features = self.raw[:, :-1]
        self.raw_labels = (self.raw[:, -1] != 0).astype(np.int64)
        self.n_timestamps = int(self.raw.shape[0])

        if self.n_timestamps < 100:
            raise ValueError("Dataset is too short for the paper protocol.")
        self._split_indices()
        self._fit_preprocessing()
        self._build_window_sets()
        self.assert_no_test_leakage()

    # ------------------------------------------------------------------ splits
    def _split_indices(self) -> None:
        split = self.config.split
        n = self.n_timestamps
        n_train = int(np.floor(n * split.train))
        n_val = int(np.floor(n * split.validation))
        n_test = n - n_train - n_val
        _require(n_train > 0 and n_val > 0 and n_test > 0, "Empty split.")

        self.train_span = (0, n_train)
        self.validation_span = (n_train, n_train + n_val)
        self.test_span = (n_train + n_val, n)

        n_fit = int(np.floor(n_train * (1.0 - self.config.predictor.val_fraction)))
        _require(n_fit > 0 and n_train - n_fit > 0,
                 "Train segment is too short for fitting and early stopping.")
        self.fit_span = (0, n_fit)
        self.predictor_val_span = (n_fit, n_train)

        val_start, val_end = self.validation_span
        n_stream = int(np.floor((val_end - val_start) * split.rl_stream_fraction))
        _require(n_stream > 0 and (val_end - val_start) - n_stream > 0,
                 "Validation segment is too short for drift stream + accuracy holdout.")
        self.stream_span = (val_start, val_start + n_stream)
        self.accuracy_span = (val_start + n_stream, val_end)

    # ----------------------------------------------------------- preprocessing
    def _fit_preprocessing(self) -> None:
        fit_start, fit_end = self.fit_span
        fit_x = self.raw_features[fit_start:fit_end]
        fit_y = self.raw_labels[fit_start:fit_end]
        if np.unique(fit_y).size < 2:
            warnings.warn(
                "The fit split contains a single class; RFE will use an "
                "unsupervised variance fallback.",
                RuntimeWarning,
            )

        self.scaler = StandardScaler().fit(fit_x)
        fit_scaled = self.scaler.transform(fit_x)
        n_features = fit_scaled.shape[1]
        wanted = min(int(self.config.predictor.rfe_features), n_features)

        self.rfe_estimator = None
        if wanted >= n_features:
            self.selected_indices = np.arange(n_features, dtype=np.int64)
            self.rfe_support = np.ones(n_features, dtype=bool)
            self.rfe_ranking = np.ones(n_features, dtype=np.int64)
        else:
            if np.unique(fit_y).size < 2:
                selector = SelectKBest(score_func=f_classif, k=wanted)
                selector.fit(fit_scaled, fit_y)
                self.selected_indices = np.flatnonzero(selector.get_support())
                self.rfe_support = selector.get_support()
                self.rfe_ranking = None
            else:
                try:
                    estimator = LogisticRegression(
                        max_iter=3_000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=self.config.seed,
                    )
                    selector = RFE(estimator, n_features_to_select=wanted, step=1)
                    selector.fit(fit_scaled, fit_y)
                    self.rfe_estimator = selector
                    self.selected_indices = np.flatnonzero(selector.support_)
                    self.rfe_support = selector.support_
                    self.rfe_ranking = selector.ranking_
                except Exception as exc:  # numerical fallback, still train-only
                    warnings.warn(
                        f"RFE failed ({exc}); falling back to ANOVA F-score on "
                        "the fit split only.",
                        RuntimeWarning,
                    )
                    selector = SelectKBest(score_func=f_classif, k=wanted)
                    selector.fit(fit_scaled, fit_y)
                    self.selected_indices = np.flatnonzero(selector.get_support())
                    self.rfe_support = selector.get_support()
                    self.rfe_ranking = None

        self.selected_indices = np.asarray(self.selected_indices, dtype=np.int64)
        self.selected_feature_names = [
            self.feature_names[i] for i in self.selected_indices
        ]
        # Transform the full timeline once with train-only statistics, but
        # never let the selected test indices reach adaptation code.
        self.features_scaled = self.scaler.transform(self.raw_features)[
            :, self.selected_indices
        ].astype(np.float32)

    # -------------------------------------------------------------- windows
    def _build_window_sets(self) -> None:
        p = self.config.predictor
        args = dict(
            input_window=p.input_window,
            forecast_horizon=p.forecast_horizon,
            timestamps=self.timestamp_values,
        )
        self.fit_windows = build_windows(
            self.features_scaled, self.raw_labels, *self.fit_span, **args
        )
        self.predictor_val_windows = build_windows(
            self.features_scaled, self.raw_labels, *self.predictor_val_span, **args
        )
        self.stream_windows = build_windows(
            self.features_scaled, self.raw_labels, *self.stream_span, **args
        )
        self.accuracy_windows = build_windows(
            self.features_scaled, self.raw_labels, *self.accuracy_span, **args
        )
        self.test_windows = build_windows(
            self.features_scaled, self.raw_labels, *self.test_span, **args
        )
        for name, windows in (
            ("fit", self.fit_windows),
            ("predictor-validation", self.predictor_val_windows),
            ("drift stream", self.stream_windows),
            ("accuracy holdout", self.accuracy_windows),
            ("test", self.test_windows),
        ):
            if windows.empty():
                raise ValueError(
                    f"Segment {name!r} is too short to form a single "
                    f"{p.input_window}+{p.forecast_horizon} window."
                )
        self._assert_no_index_overlap()

    # -------------------------------------------------------------- leakage
    def _assert_no_index_overlap(self) -> None:
        def spans(windows: WindowSet, horizon: int):
            # Occupied timestamps: input plus forecast target.
            return set(
                range(int(windows.starts.min()),
                      int(windows.starts.max()) + self.config.predictor.input_window + horizon)
            )

        horizon = self.config.predictor.forecast_horizon
        spans_by_name = {
            "fit": spans(self.fit_windows, horizon),
            "predictor-validation": spans(self.predictor_val_windows, horizon),
            "drift stream": spans(self.stream_windows, horizon),
            "accuracy holdout": spans(self.accuracy_windows, horizon),
            "test": spans(self.test_windows, horizon),
        }
        names = list(spans_by_name)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                overlap = spans_by_name[left] & spans_by_name[right]
                if overlap:
                    raise AssertionError(
                        f"Leakage: {left} and {right} share timeline indices."
                    )

    def assert_no_test_leakage(self) -> None:
        """Fail loudly if the test span intersects any training/RL span."""
        test_start, test_end = self.test_span
        protected = (
            self.fit_span,
            self.predictor_val_span,
            self.stream_span,
            self.accuracy_span,
        )
        for start, end in protected:
            if not (end <= test_start or start >= test_end):
                raise AssertionError("Test span overlaps an adaptation span.")

    # -------------------------------------------------------------- helpers
    def summary(self) -> Dict[str, object]:
        def count(windows: WindowSet) -> Dict[str, object]:
            return {
                "windows": len(windows),
                "positive": int(windows.labels.sum()),
                "positive_rate": float(windows.labels.mean()),
                "start": int(windows.starts.min()),
                "end": int(windows.starts.max()),
            }

        return {
            "data_root": str(self.data_root),
            "timestamps": self.n_timestamps,
            "raw_features": len(self.feature_names),
            "selected_features": len(self.selected_indices),
            "selected_feature_names": self.selected_feature_names,
            "spans": {
                "train": self.train_span,
                "fit": self.fit_span,
                "predictor_val": self.predictor_val_span,
                "validation": self.validation_span,
                "drift_stream": self.stream_span,
                "accuracy_holdout": self.accuracy_span,
                "test": self.test_span,
            },
            "windows": {
                "fit": count(self.fit_windows),
                "predictor_val": count(self.predictor_val_windows),
                "drift_stream": count(self.stream_windows),
                "accuracy_holdout": count(self.accuracy_windows),
                "test": count(self.test_windows),
            },
        }

    def save_preprocessing(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.scaler.mean_,
            scale=self.scaler.scale_,
            var=self.scaler.var_,
            selected_indices=self.selected_indices,
            feature_names=np.asarray(self.feature_names, dtype=object),
            selected_feature_names=np.asarray(self.selected_feature_names, dtype=object),
        )
