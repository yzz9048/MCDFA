"""Anomaly scoring and reports for LSTM encoder-decoder reconstruction.

The protocol follows Malhotra et al. (2016): reconstruction errors on normal
validation windows estimate a Gaussian ``N(mu, Sigma)``; the anomaly score of a
point is its squared Mahalanobis distance to that distribution.  A threshold is
selected on validation points to maximize F_beta (beta < 1 by default), then
applied unchanged to the test split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _as_errors(errors):
    errors = np.asarray(errors, dtype=np.float64)
    if errors.ndim < 2:
        raise ValueError('Reconstruction errors must have shape [..., features].')
    if errors.shape[-1] < 1:
        raise ValueError('Reconstruction errors must contain at least one feature.')
    return errors


def fit_gaussian(errors, regularization=1.0e-3):
    """Estimate ``mu`` and ``Sigma`` from normal reconstruction errors (MLE)."""
    errors = _as_errors(errors)
    flat = errors.reshape(-1, errors.shape[-1])
    if flat.shape[0] < 1:
        raise ValueError('Cannot fit an error distribution without samples.')
    if not np.isfinite(flat).all():
        raise ValueError('Reconstruction errors must be finite.')

    mean = flat.mean(axis=0)
    if flat.shape[0] == 1:
        covariance = np.eye(flat.shape[1], dtype=np.float64)
    else:
        covariance = np.atleast_2d(np.cov(flat, rowvar=False, bias=True))
    if covariance.shape != (flat.shape[1], flat.shape[1]):
        raise ValueError('Covariance estimation failed for the error distribution.')
    if not np.isfinite(covariance).all():
        covariance = np.eye(flat.shape[1], dtype=np.float64)
    regularization = float(regularization)
    if regularization < 0 or not np.isfinite(regularization):
        raise ValueError('regularization must be finite and non-negative.')
    covariance = covariance + regularization * np.eye(flat.shape[1])

    return {
        'mean': mean,
        'covariance': covariance,
        'inverse_covariance': np.linalg.pinv(covariance),
        'sample_count': int(flat.shape[0]),
        'regularization': regularization,
    }


def mahalanobis_scores(errors, stats):
    """Squared Mahalanobis distance of each point's error vector."""
    errors = _as_errors(errors)
    mean = np.asarray(stats['mean'], dtype=np.float64)
    inverse = np.asarray(stats['inverse_covariance'], dtype=np.float64)
    features = errors.shape[-1]
    if mean.shape != (features,) or inverse.shape != (features, features):
        raise ValueError('Error distribution does not match the feature dimension.')

    flat = errors.reshape(-1, features)
    difference = flat - mean
    scores = np.einsum('ni,ij,nj->n', difference, inverse, difference)
    scores = np.maximum(scores, 0.0)
    return scores.reshape(errors.shape[:-1])


def threshold_candidates(scores, configured_threshold=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = scores[np.isfinite(scores)]
    thresholds = set()
    if finite.size:
        quantiles = np.quantile(finite, np.linspace(0.0, 1.0, 101))
        thresholds.update(float(value) for value in quantiles if np.isfinite(value))
        thresholds.add(float(np.nextafter(np.max(finite), np.inf)))
    if (
        configured_threshold is not None
        and np.isfinite(float(configured_threshold))
        and float(configured_threshold) > 0
    ):
        thresholds.add(float(configured_threshold))
    return sorted(threshold for threshold in thresholds if np.isfinite(threshold))


def _classification_metrics(scores, labels, threshold, beta):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1).astype(bool)
    predicted = scores >= float(threshold)
    true_positive = int(np.count_nonzero(predicted & labels))
    false_positive = int(np.count_nonzero(predicted & ~labels))
    false_negative = int(np.count_nonzero(~predicted & labels))
    true_negative = int(np.count_nonzero(~predicted & ~labels))

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    beta2 = float(beta) ** 2
    fbeta = (
        (1.0 + beta2) * precision * recall / (beta2 * precision + recall)
        if beta2 * precision + recall > 0 else 0.0
    )
    return {
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fbeta': float(fbeta),
        'true_positive': true_positive,
        'false_positive': false_positive,
        'false_negative': false_negative,
        'true_negative': true_negative,
        'predicted_positive': int(np.count_nonzero(predicted)),
        'actual_positive': int(np.count_nonzero(labels)),
        'count': int(labels.size),
    }


def compute_point_metrics(point_scores, point_labels, threshold, beta=0.05):
    return _classification_metrics(point_scores, point_labels, threshold, beta)


def compute_window_metrics(
    point_scores, point_labels, threshold, window_labels=None, beta=0.05
):
    point_scores = np.asarray(point_scores, dtype=np.float64)
    point_labels = np.asarray(point_labels)
    if point_scores.ndim != 2 or point_scores.shape != point_labels.shape:
        raise ValueError('Point scores and labels must share shape [windows, length].')
    predicted = (point_scores >= float(threshold)).any(axis=1)
    if window_labels is None:
        true = (point_labels != 0).any(axis=1)
    else:
        true = np.asarray(window_labels).reshape(-1).astype(bool)
        if true.size != predicted.size:
            raise ValueError('window_labels must contain one value per window.')
    metrics = _classification_metrics(
        predicted.astype(np.float64), true, 0.5, beta
    )
    metrics['threshold'] = float(threshold)
    return metrics


def select_threshold(
    point_scores,
    point_labels,
    beta=0.05,
    configured_threshold=None,
    fallback_false_positive_rate=0.01,
):
    """Select a point threshold on validation data by maximizing F_beta."""
    scores = np.asarray(point_scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(point_labels).reshape(-1).astype(bool)
    if scores.size != labels.size:
        raise ValueError('Point scores and labels must have the same length.')
    if not np.isfinite(scores).all():
        raise ValueError('Point scores must be finite.')

    beta = float(beta)
    if beta <= 0 or not np.isfinite(beta):
        raise ValueError('beta must be finite and positive.')

    if not labels.any():
        fallback = float(fallback_false_positive_rate)
        if not 0 < fallback <= 1 or not np.isfinite(fallback):
            raise ValueError('fallback_false_positive_rate must be in (0, 1].')
        threshold = float(np.quantile(scores, 1.0 - fallback))
        if not np.isfinite(threshold):
            threshold = float(np.nextafter(np.max(scores), np.inf))
        metrics = _classification_metrics(scores, labels, threshold, beta)
        return {
            'objective': 'fbeta',
            'beta': beta,
            'selected_threshold': threshold,
            'selected_metrics': metrics,
            'selection_rule': 'fallback_false_positive_rate (no validation positives)',
            'configured_threshold': configured_threshold,
            'fallback_false_positive_rate': fallback,
            'sweep': [metrics],
        }

    sweep = [
        _classification_metrics(scores, labels, threshold, beta)
        for threshold in threshold_candidates(scores, configured_threshold)
    ]
    if not sweep:
        raise ValueError('No finite threshold candidates were produced.')

    def key(row):
        proximity = (
            -abs(row['threshold'] - float(configured_threshold))
            if configured_threshold is not None
            and np.isfinite(float(configured_threshold))
            else 0.0
        )
        return (
            row['fbeta'],
            row['recall'],
            row['precision'],
            proximity,
            row['threshold'],
        )

    selected = max(sweep, key=key)
    return {
        'objective': 'fbeta',
        'beta': beta,
        'selected_threshold': selected['threshold'],
        'selected_metrics': selected,
        'selection_rule': (
            'maximize validation F_beta; ties favor recall, precision, '
            'proximity to the configured threshold, then a higher threshold'
        ),
        'configured_threshold': configured_threshold,
        'fallback_false_positive_rate': float(fallback_false_positive_rate),
        'sweep': sweep,
    }


def window_scores_from_points(point_scores, reduction='max'):
    point_scores = np.asarray(point_scores, dtype=np.float64)
    if point_scores.ndim != 2:
        raise ValueError('Point scores must have shape [windows, length].')
    if reduction == 'max':
        return point_scores.max(axis=1)
    if reduction == 'mean':
        return point_scores.mean(axis=1)
    if reduction == 'topk_mean':
        k = max(1, int(np.ceil(point_scores.shape[1] * 0.1)))
        return np.partition(point_scores, -k, axis=1)[:, -k:].mean(axis=1)
    raise ValueError('window_score_reduction must be max, mean, or topk_mean.')


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )


def write_reconstruction_report(
    dataset,
    point_scores,
    threshold,
    beta=0.05,
    output_dir='.',
    split=None,
    config=None,
    stats=None,
    checkpoint=None,
    checkpoint_step=None,
    threshold_selection=None,
):
    """Save point/window scores, threshold selection, and a JSON summary."""
    point_scores = np.asarray(point_scores, dtype=np.float64)
    point_labels = np.asarray(dataset.labels)
    window_labels = np.asarray(dataset.window_labels, dtype=np.int64)
    starts = np.asarray(dataset.sample_starts, dtype=np.int64)
    split = split or dataset.period

    if point_scores.ndim != 2 or point_scores.shape != point_labels.shape:
        raise ValueError('Point scores must share shape [windows, length] with labels.')
    if len(point_scores) != len(starts) or len(point_scores) != len(window_labels):
        raise ValueError('Every window must have one score set and one label.')
    if not len(point_scores):
        raise ValueError('Cannot report an empty reconstruction split.')
    if not np.isfinite(point_scores).all():
        raise ValueError('Point scores must be finite.')
    if dataset.period not in {'validation', 'test'}:
        raise ValueError('Reconstruction reports require a validation or test split.')
    if np.any(np.diff(starts) <= 0):
        raise ValueError('Reconstruction windows must be in chronological order.')

    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError('The anomaly threshold must be finite and non-negative.')

    config = config or {}
    reduction = config.get('window_score_reduction', 'max')
    window_scores = window_scores_from_points(point_scores, reduction)
    window_prediction = (point_scores >= threshold).any(axis=1)

    point_metrics = compute_point_metrics(point_scores, point_labels, threshold, beta)
    window_metrics = compute_window_metrics(
        point_scores, point_labels, threshold, window_labels=window_labels, beta=beta
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_end = starts + dataset.window - 1
    timestamps = np.asarray(dataset.raw_timestamps)
    scores = pd.DataFrame({
        'split': split,
        'window_start': starts,
        'window_end': window_end,
        'start_timestamp': timestamps[starts],
        'end_timestamp': timestamps[window_end],
        'window_label': window_labels,
        'anomaly_score': window_scores,
        'max_point_score': point_scores.max(axis=1),
        'mean_point_score': point_scores.mean(axis=1),
        'prediction': window_prediction.astype(np.int64),
        'decision_threshold': threshold,
    })
    scores.to_csv(output_dir / f'{dataset.name}_window_scores_{split}.csv', index=False)

    if config.get('save_point_scores', False):
        point_positions = starts[:, None] + np.arange(dataset.window)[None, :]
        np.savez_compressed(
            output_dir / f'{dataset.name}_point_scores_{split}.npz',
            point_scores=point_scores,
            point_labels=point_labels,
            window_starts=starts,
            point_timestamps=timestamps[point_positions],
            decision_threshold=np.asarray([threshold]),
        )

    if threshold_selection is not None:
        pd.DataFrame(threshold_selection['sweep']).to_csv(
            output_dir / f'{dataset.name}_threshold_sweep_validation.csv',
            index=False,
        )
        _write_json(
            output_dir / f'{dataset.name}_threshold_selection.json',
            threshold_selection,
        )

    distribution = None
    if stats is not None:
        distribution = {
            'sample_count': stats.get('sample_count'),
            'regularization': stats.get('regularization'),
            'mean': np.asarray(stats['mean']).tolist(),
            'covariance': np.asarray(stats['covariance']).tolist(),
        }

    summary = {
        'dataset': dataset.name,
        'split': split,
        'task': 'reconstruction',
        'label_scope': 'window_contains_fault',
        'checkpoint': checkpoint,
        'checkpoint_step': checkpoint_step,
        'window': dataset.window,
        'stride': dataset.DeltaT,
        'train_target_end_exclusive': dataset.split_timestamp,
        'validation_target_range': [dataset.split_timestamp, dataset.validation_timestamp],
        'test_target_range': [dataset.validation_timestamp, dataset.len],
        'split_window_counts': dataset.split_window_counts,
        'boundary_crossing_windows_excluded': dataset.boundary_crossing_windows_excluded,
        'threshold': threshold,
        'beta': float(beta),
        'window_score_reduction': reduction,
        'validation_threshold_selection': threshold_selection,
        'metrics': {'point': point_metrics, 'window': window_metrics},
        'error_distribution': distribution,
    }
    _write_json(output_dir / f'{dataset.name}_detection_summary_{split}.json', summary)

    selected_path = output_dir / f'{dataset.name}_selected_threshold_metrics.csv'
    selected_rows = []
    if split == 'test' and selected_path.is_file():
        previous = pd.read_csv(selected_path)
        selected_rows = previous[previous['split'] == 'validation'].to_dict('records')
    row = {
        'split': split,
        'threshold': threshold,
        'beta': float(beta),
        'precision': window_metrics['precision'],
        'recall': window_metrics['recall'],
        'f1': window_metrics['f1'],
        'fbeta': window_metrics['fbeta'],
        'true_positive': window_metrics['true_positive'],
        'false_positive': window_metrics['false_positive'],
        'false_negative': window_metrics['false_negative'],
        'true_negative': window_metrics['true_negative'],
        'window_count': window_metrics['count'],
        'positive_windows': window_metrics['actual_positive'],
        'predicted_positive_windows': window_metrics['predicted_positive'],
    }
    pd.DataFrame(selected_rows + [row]).to_csv(selected_path, index=False)

    return summary
