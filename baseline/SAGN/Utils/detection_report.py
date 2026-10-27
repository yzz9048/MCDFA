"""Validation threshold selection and event reports using the CDFA-bk protocol."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from Utils.lead_time import compute_event_lead_time

def alert_score(row):
    return float(row.get('alert_probability', row['fault_probability']))

def onset_episodes(raw_labels):
    fault = np.asarray(raw_labels) != 0
    onset_mask = np.zeros_like(fault, dtype=bool)
    onset_mask[1:] = fault[1:] & ~fault[:-1]
    starts = np.flatnonzero(onset_mask)
    episodes = []
    for onset in starts:
        end = onset
        while end + 1 < len(fault) and fault[end + 1]:
            end += 1
        episodes.append((int(onset), int(end)))
    return episodes

def threshold_candidates(rows, raw_labels, configured_threshold):
    """Sweep the observed score range and exact forecastable-onset boundaries.

    Alert scores are usually probabilities, but causal robust-deviation scores
    are unbounded. A fixed 0..1 sweep silently makes those scores fire on nearly
    every window, so derive a coarse quantile grid from the validation scores.
    """
    scores = np.asarray([alert_score(row) for row in rows], dtype=float)
    finite_scores = scores[np.isfinite(scores)]
    thresholds = set()
    if finite_scores.size:
        quantiles = np.quantile(finite_scores, np.linspace(0.0, 1.0, 101))
        thresholds.update(float(value) for value in quantiles if value > 0.0)
        thresholds.add(float(np.nextafter(np.max(finite_scores), np.inf)))
    else:
        thresholds.update(np.round(np.arange(0.01, 1.0, 0.01), 2).tolist())
    if np.isfinite(float(configured_threshold)) and float(configured_threshold) > 0:
        thresholds.add(float(configured_threshold))

    # Event recall and lead time only change when a window that covers an onset
    # crosses the threshold. Include those observed score values so the event
    # objective is not limited to 0.01-wide steps.
    onset_indices = [onset for onset, _ in onset_episodes(raw_labels)]
    event_scores = set()
    for row in rows:
        obs_end = int(row['observation_end'])
        target_end = int(row['target_end'])
        if any(obs_end < onset <= target_end for onset in onset_indices):
            score = float(alert_score(row))
            if np.isfinite(score) and score > 0.0:
                event_scores.add(score)
    thresholds.update(event_scores)
    # Capture the state immediately above each event-relevant score too.
    thresholds.update(
        float(np.nextafter(score, np.inf)) for score in event_scores
    )
    return sorted(
        threshold for threshold in thresholds
        if np.isfinite(threshold) and threshold > 0.0
    )

def event_metrics(rows, raw_labels, timestamps, threshold, split_name, keep_details=False):
    obs_ends = np.asarray([row['observation_end'] for row in rows], dtype=np.int64)
    target_ends = np.asarray([row['target_end'] for row in rows], dtype=np.int64)
    probs = np.asarray([alert_score(row) for row in rows], dtype=float)
    positive = probs >= threshold
    episodes = onset_episodes(raw_labels)
    details = []
    pre_probability_rows = []
    eligible_onset_indices = []
    eligible = detected = alerted = post_only = 0
    lead_times = []

    for onset, episode_end in episodes:
        coverage = (obs_ends < onset) & (target_ends >= onset)
        if not coverage.any():
            continue
        eligible += 1
        eligible_onset_indices.append(onset)
        pre_alert_mask = coverage & positive
        pre_alert_indices = np.flatnonzero(pre_alert_mask)
        for idx in np.flatnonzero(coverage):
            row = rows[int(idx)]
            pre_probability_rows.append({
                'split': split_name,
                'onset_index': onset,
                'onset_timestamp': str(timestamps[onset]),
                'window_start': row['window_start'],
                'observation_end': row['observation_end'],
                'observation_timestamp': row['observation_timestamp'],
                'target_end': row['target_end'],
                'fault_probability': row['fault_probability'],
                'onset_probability': row.get('onset_probability'),
                'lead_weighted_probability': row.get('lead_weighted_probability'),
                'alert_probability': alert_score(row),
                'threshold': threshold,
                'above_threshold': bool(alert_score(row) >= threshold),
            })

        # Identify the first alert in the lead-up or during this fault episode.
        near_episode = (
            (obs_ends >= onset - (target_ends - obs_ends))
            & (obs_ends <= episode_end)
            & positive
        )
        alarm_indices = np.flatnonzero(near_episode)
        first_alarm = int(alarm_indices[0]) if alarm_indices.size else None
        detected_before = bool(pre_alert_indices.size)
        if detected_before:
            first_pre_idx = int(pre_alert_indices[0])
            lead = int(onset - obs_ends[first_pre_idx])
            detected += 1
            lead_times.append(lead)
        else:
            lead = None
        if first_alarm is not None:
            alerted += 1
            if not detected_before:
                post_only += 1

        details.append({
            'split': split_name,
            'onset_index': onset,
            'onset_timestamp': str(timestamps[onset]),
            'fault_end_index': episode_end,
            'fault_end_timestamp': str(timestamps[episode_end]),
            'eligible_for_forecast': True,
            'first_alarm_observation_end': (
                int(obs_ends[first_alarm]) if first_alarm is not None else None
            ),
            'first_alarm_timestamp': (
                str(timestamps[obs_ends[first_alarm]]) if first_alarm is not None else None
            ),
            'first_alarm_probability': (
                float(probs[first_alarm]) if first_alarm is not None else None
            ),
            'first_pre_onset_alarm_observation_end': (
                int(obs_ends[first_pre_idx]) if detected_before else None
            ),
            'first_pre_onset_alarm_timestamp': (
                str(timestamps[obs_ends[first_pre_idx]]) if detected_before else None
            ),
            'first_pre_onset_probability': (
                float(probs[first_pre_idx]) if detected_before else None
            ),
            'lead_time_timestamps': lead,
            'detected_before_onset': detected_before,
            'alarm_only_after_onset': bool(first_alarm is not None and not detected_before),
            'threshold': threshold,
        })

    # Collapse adjacent alarm windows (allowing one missing stride) into alert
    # episodes. Match each episode to at most one future onset so a persistent
    # alert cannot count as many separate event alarms.
    alert_indices = np.flatnonzero(positive)
    unique_obs = np.unique(obs_ends)
    stride = int(np.min(np.diff(unique_obs))) if unique_obs.size > 1 else 1
    merge_gap = 2 * stride
    alarm_clusters = []
    for idx in alert_indices:
        if (
            alarm_clusters
            and obs_ends[int(idx)] - obs_ends[alarm_clusters[-1][-1]] <= merge_gap
        ):
            alarm_clusters[-1].append(int(idx))
        else:
            alarm_clusters.append([int(idx)])

    matched_onsets = set()
    matched_alarm_events = 0
    matched_event_leads = []
    alarm_event_details = []
    for alarm_number, cluster in enumerate(alarm_clusters, start=1):
        candidate_onsets = []
        first_matching_index = {}
        for onset in eligible_onset_indices:
            matching = [
                idx for idx in cluster
                if obs_ends[idx] < onset <= target_ends[idx]
            ]
            if matching:
                candidate_onsets.append(onset)
                first_matching_index[onset] = matching[0]
        available_candidates = [
            onset for onset in candidate_onsets if onset not in matched_onsets
        ]
        matched_onset = min(available_candidates) if available_candidates else None
        if matched_onset is not None:
            matched_onsets.add(matched_onset)
            matched_alarm_events += 1
            match_index = first_matching_index[matched_onset]
            matched_event_leads.append(int(matched_onset - obs_ends[match_index]))
        else:
            match_index = None
        first_idx, last_idx = cluster[0], cluster[-1]
        alarm_event_details.append({
            'split': split_name,
            'alarm_event_id': alarm_number,
            'first_alarm_observation_end': int(obs_ends[first_idx]),
            'first_alarm_timestamp': str(timestamps[obs_ends[first_idx]]),
            'last_alarm_observation_end': int(obs_ends[last_idx]),
            'last_alarm_timestamp': str(timestamps[obs_ends[last_idx]]),
            'alarm_window_count': len(cluster),
            'max_alert_probability': float(max(probs[idx] for idx in cluster)),
            'matched_onset_index': matched_onset,
            'matched_onset_timestamp': (
                str(timestamps[matched_onset]) if matched_onset is not None else None
            ),
            'lead_time_timestamps': (
                int(matched_onset - obs_ends[match_index])
                if matched_onset is not None else None
            ),
            'is_false_alarm_event': matched_onset is None,
            'threshold': threshold,
        })

    event_precision = (
        matched_alarm_events / len(alarm_clusters) if alarm_clusters else 0.0
    )
    event_recall = (
        matched_alarm_events / eligible if eligible else float('nan')
    )
    event_f1 = (
        2.0 * event_precision * event_recall / (event_precision + event_recall)
        if np.isfinite(event_recall) and event_precision + event_recall > 0
        else 0.0
    )
    for detail in details:
        detail['matched_by_event_level_alarm'] = (
            detail['onset_index'] in matched_onsets
        )

    return {
        'eligible_onsets': eligible,
        'detected_onsets': detected,
        'event_detection_rate': detected / eligible if eligible else float('nan'),
        'event_precision': event_precision,
        'event_recall': event_recall,
        'event_f1': event_f1,
        'predicted_alarm_events': len(alarm_clusters),
        'matched_alarm_events': matched_alarm_events,
        'false_alarm_events': len(alarm_clusters) - matched_alarm_events,
        'event_avg_lead_time_timestamps': (
            float(np.mean(matched_event_leads))
            if matched_event_leads else float('nan')
        ),
        'event_avg_lead_time_all_eligible_onsets_timestamps': (
            float(np.sum(matched_event_leads) / eligible) if eligible else float('nan')
        ),
        'avg_lead_time_timestamps': float(np.mean(lead_times)) if lead_times else float('nan'),
        'avg_lead_time_all_eligible_onsets_timestamps': (
            float(np.sum(lead_times) / eligible) if eligible else float('nan')
        ),
        'onsets_with_any_alarm': alerted,
        'post_onset_only_onsets': post_only,
        'post_onset_only_ratio_among_alerted_onsets': post_only / alerted if alerted else float('nan'),
        'event_details': details if keep_details else None,
        'alarm_event_details': alarm_event_details if keep_details else None,
        'pre_onset_window_probabilities': pre_probability_rows if keep_details else None,
    }

def metric_row(rows, threshold, raw_labels, timestamps, split_name):
    y_true = np.asarray([row['window_label'] for row in rows], dtype=int)
    probabilities = np.asarray([alert_score(row) for row in rows], dtype=float)
    predicted = (probabilities >= threshold).astype(int)
    events = event_metrics(rows, raw_labels, timestamps, threshold, split_name)
    post_window_alerts = 0
    alert_windows = int(predicted.sum())
    for row, is_alert in zip(rows, predicted):
        if is_alert and raw_labels[row['observation_end']] != 0:
            post_window_alerts += 1
    point_f1 = float(f1_score(y_true, predicted, zero_division=0))
    event_f1 = float(events['event_f1'])
    balanced_f1 = (
        2.0 * point_f1 * event_f1 / (point_f1 + event_f1)
        if point_f1 + event_f1 > 0.0 else 0.0
    )
    return {
        'split': split_name,
        'threshold': float(threshold),
        'precision': float(precision_score(y_true, predicted, zero_division=0)),
        'window_recall': float(recall_score(y_true, predicted, zero_division=0)),
        'f1': point_f1,
        'balanced_f1': balanced_f1,
        'window_count': int(len(rows)),
        'positive_windows': int(y_true.sum()),
        'predicted_positive_windows': alert_windows,
        'false_positive_windows': int(((predicted == 1) & (y_true == 0)).sum()),
        'post_onset_alert_windows': post_window_alerts,
        'post_onset_alert_window_ratio': post_window_alerts / alert_windows if alert_windows else float('nan'),
        'eligible_onsets': events['eligible_onsets'],
        'detected_onsets': events['detected_onsets'],
        'event_detection_rate': events['event_detection_rate'],
        'event_precision': events['event_precision'],
        'event_recall': events['event_recall'],
        'event_f1': events['event_f1'],
        'predicted_alarm_events': events['predicted_alarm_events'],
        'matched_alarm_events': events['matched_alarm_events'],
        'total_matched_lead_time_timestamps': (
            events['event_avg_lead_time_all_eligible_onsets_timestamps'] * events['eligible_onsets']
            if events['eligible_onsets'] else 0.0
        ),
        'alarm_window_ratio': alert_windows / len(rows),
        'false_alarm_events': events['false_alarm_events'],
        'event_avg_lead_time_timestamps': events['event_avg_lead_time_timestamps'],
        'event_avg_lead_time_all_eligible_onsets_timestamps': events[
            'event_avg_lead_time_all_eligible_onsets_timestamps'
        ],
        'avg_lead_time_timestamps': events['avg_lead_time_timestamps'],
        'avg_lead_time_all_eligible_onsets_timestamps': events[
            'avg_lead_time_all_eligible_onsets_timestamps'
        ],
        'onsets_with_any_alarm': events['onsets_with_any_alarm'],
        'post_onset_only_onsets': events['post_onset_only_onsets'],
        'post_onset_only_ratio_among_alerted_onsets': events['post_onset_only_ratio_among_alerted_onsets'],
    }


def select_validation_threshold(rows, raw_labels, timestamps, config):
    """Select only from validation scores, with the reference tie-break rules."""
    if not rows or any(row['split'] != 'validation' for row in rows):
        raise ValueError('Threshold selection requires non-empty validation-only rows.')
    configured_threshold = float(config.get('decision_threshold', 0.5))
    objective = config.get('threshold_selection_objective', 'balanced')
    if objective not in {'balanced', 'event', 'hits_lead'}:
        raise ValueError('threshold_selection_objective must be balanced, event, or hits_lead.')
    minimum_recall = float(config.get('min_validation_event_recall', 0.0))
    minimum_lead = float(config.get('min_validation_event_lead_time', 0.0))
    if not 0 <= minimum_recall <= 1 or not np.isfinite(minimum_lead) or minimum_lead < 0:
        raise ValueError('Invalid validation event recall or lead-time floor.')
    sweep = [
        metric_row(rows, threshold, raw_labels, timestamps, 'validation')
        for threshold in threshold_candidates(rows, raw_labels, configured_threshold)
    ]
    eligible = sweep
    floors_relaxed = False
    if objective == 'balanced':
        eligible = [row for row in sweep if (
            np.isfinite(row['event_recall'])
            and row['event_recall'] >= minimum_recall
            and np.isfinite(row['event_avg_lead_time_all_eligible_onsets_timestamps'])
            and row['event_avg_lead_time_all_eligible_onsets_timestamps'] >= minimum_lead
        )]
        if not eligible:
            eligible = sweep
            floors_relaxed = True

    def finite(value):
        return value if np.isfinite(value) else -1.0

    if objective == 'hits_lead':
        key = lambda row: (
            row['matched_alarm_events'], row['total_matched_lead_time_timestamps'],
            finite(row['event_avg_lead_time_timestamps']),
            -abs(row['threshold'] - configured_threshold), row['threshold'],
        )
        rule = ('maximize validation one-to-one matched onset count, then total matched lead time; '
                'ties favor matched mean lead time, proximity to the configured threshold, '
                'then a higher threshold; event precision/recall/F1 are not optimization criteria')
    elif objective == 'balanced':
        key = lambda row: (
            row['balanced_f1'],
            finite(row['event_avg_lead_time_all_eligible_onsets_timestamps']),
            finite(row['event_recall']), row['event_f1'], row['f1'],
            -row['false_alarm_events'], row['threshold'],
        )
        rule = ('maximize the harmonic mean of window F1 and event F1 on validation; '
                'ties favor all-eligible lead time, event recall, event F1, '
                'window F1, fewer false alarm events, then a higher threshold')
    else:
        key = lambda row: (
            finite(row['event_recall']),
            finite(row['event_avg_lead_time_all_eligible_onsets_timestamps']),
            row['event_f1'], row['event_precision'], -row['false_alarm_events'],
            finite(row['event_detection_rate']),
            finite(row['event_avg_lead_time_timestamps']),
            finite(row['avg_lead_time_all_eligible_onsets_timestamps']),
            finite(row['avg_lead_time_timestamps']), row['f1'], row['window_recall'],
            -row['false_positive_windows'], row['threshold'],
        )
        rule = ('maximize validation event recall, then all-eligible lead time, '
                'event F1 and precision, fewer false alarm events, and window metrics')
    selected = max(eligible, key=key)
    return {
        'objective': objective,
        'min_validation_event_recall': minimum_recall,
        'min_validation_event_lead_time': minimum_lead,
        'selected_threshold': selected['threshold'],
        'selected_validation_metrics': selected,
        'selection_rule': rule,
        'event_floors_relaxed': floors_relaxed,
    }, sweep


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _write_json(path, payload):
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


ONSET_COLUMNS = [
    'split', 'onset_index', 'onset_timestamp', 'fault_end_index', 'fault_end_timestamp',
    'eligible_for_forecast', 'first_alarm_observation_end', 'first_alarm_timestamp',
    'first_alarm_probability', 'first_pre_onset_alarm_observation_end',
    'first_pre_onset_alarm_timestamp', 'first_pre_onset_probability',
    'lead_time_timestamps', 'detected_before_onset', 'alarm_only_after_onset',
    'threshold', 'matched_by_event_level_alarm',
]
ALARM_COLUMNS = [
    'split', 'alarm_event_id', 'first_alarm_observation_end', 'first_alarm_timestamp',
    'last_alarm_observation_end', 'last_alarm_timestamp', 'alarm_window_count',
    'max_alert_probability', 'matched_onset_index', 'matched_onset_timestamp',
    'lead_time_timestamps', 'is_false_alarm_event', 'threshold',
]
PRE_ONSET_COLUMNS = [
    'split', 'onset_index', 'onset_timestamp', 'window_start', 'observation_end',
    'observation_timestamp', 'target_end', 'fault_probability', 'onset_probability',
    'lead_weighted_probability', 'alert_probability', 'threshold', 'above_threshold',
]


def write_detection_report(dataset, probabilities, labels, pred_len, output_dir,
                           config, threshold=None, checkpoint=None, checkpoint_step=None):
    """Save scores, event details, and validation-only threshold selection."""
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    starts = np.asarray(dataset.sample_starts, dtype=np.int64)
    if dataset.period not in {'validation', 'test'}:
        raise ValueError('Detection reports require a validation or test dataset.')
    if len(starts) != len(probabilities) or len(starts) != len(labels) or not len(starts):
        raise ValueError('Every forecast window must have exactly one score and label.')
    if np.any(np.diff(starts) <= 0):
        raise ValueError('Detection windows must be in strictly chronological order.')
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('Fault probabilities must be finite values in [0, 1].')
    expected_labels = (dataset.labels[:, -pred_len:] != 0).any(axis=1).astype(np.int64)
    if not np.array_equal(labels, expected_labels):
        raise ValueError('Detection labels must describe only the future forecast interval.')
    observation_ends = starts + dataset.window - pred_len - 1
    target_starts = observation_ends + 1
    target_ends = observation_ends + pred_len
    lower = dataset.split_timestamp if dataset.period == 'validation' else dataset.validation_timestamp
    upper = dataset.validation_timestamp if dataset.period == 'validation' else dataset.len
    if np.any(target_starts < lower) or np.any(target_ends >= upper):
        raise ValueError('Forecast targets must stay within their chronological split.')
    scores = pd.DataFrame({
        'split': dataset.period,
        'window_start': starts,
        'observation_end': observation_ends,
        'observation_timestamp': dataset.raw_timestamps[observation_ends],
        'target_start': target_starts,
        'target_end': target_ends,
        'window_label': labels,
        'fault_probability': probabilities,
        'alert_probability': probabilities,
    })
    rows = scores.to_dict('records')
    selection = None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dataset.period == 'validation' and config.get('select_threshold_on_validation', True):
        selection, sweep = select_validation_threshold(rows, dataset.raw_labels, dataset.raw_timestamps, config)
        threshold = selection['selected_threshold']
        pd.DataFrame(sweep).to_csv(output_dir / f'{dataset.name}_threshold_sweep_validation.csv', index=False)
    if threshold is None:
        threshold = float(config.get('decision_threshold', 0.5))
    threshold = float(threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('The decision threshold must be finite and positive.')
    predictions = (probabilities >= threshold).astype(np.int64)
    scores['prediction'] = predictions
    scores['decision_threshold'] = threshold
    scores.to_csv(output_dir / f'{dataset.name}_window_scores_{dataset.period}.csv', index=False)
    metrics = metric_row(rows, threshold, dataset.raw_labels, dataset.raw_timestamps, dataset.period)
    details = event_metrics(rows, dataset.raw_labels, dataset.raw_timestamps, threshold, dataset.period, keep_details=True)
    for suffix, values, columns in (
        ('onset_report', details['event_details'], ONSET_COLUMNS),
        ('alarm_event_report', details['alarm_event_details'], ALARM_COLUMNS),
        ('onset_pre_window_probabilities', details['pre_onset_window_probabilities'], PRE_ONSET_COLUMNS),
    ):
        pd.DataFrame(values, columns=columns).to_csv(output_dir / f'{dataset.name}_{suffix}_{dataset.period}.csv', index=False)
    lead_time, detected_onsets, eligible_onsets = compute_event_lead_time(
        predictions, starts, dataset.raw_labels, dataset.window, pred_len,
    )
    summary = {
        'dataset': dataset.name,
        'split': dataset.period,
        'checkpoint': checkpoint,
        'checkpoint_step': checkpoint_step,
        'label_scope': 'forecast_interval_any_fault',
        'window': dataset.window, 'pred_len': pred_len, 'stride': dataset.DeltaT,
        'train_target_end_exclusive': dataset.split_timestamp,
        'validation_target_range': [dataset.split_timestamp, dataset.validation_timestamp],
        'test_target_range': [dataset.validation_timestamp, dataset.len],
        'split_window_counts': dataset.split_window_counts,
        'boundary_crossing_windows_excluded': dataset.boundary_crossing_windows_excluded,
        'threshold': threshold,
        'validation_threshold_selection': selection,
        'metrics': metrics,
        'one_to_one_window_lead_time': {
            'avg_lead_time_timestamps': lead_time,
            'detected_onsets': detected_onsets,
            'eligible_onsets': eligible_onsets,
        },
    }
    if selection is not None:
        _write_json(output_dir / f'{dataset.name}_threshold_selection.json', summary)
    _write_json(output_dir / f'{dataset.name}_detection_summary_{dataset.period}.json', summary)
    selected_path = output_dir / f'{dataset.name}_selected_threshold_metrics.csv'
    selected_rows = []
    if dataset.period == 'test' and selected_path.is_file():
        previous = pd.read_csv(selected_path)
        selected_rows = previous[previous['split'] == 'validation'].to_dict('records')
    pd.DataFrame(selected_rows + [metrics]).to_csv(selected_path, index=False)
    return summary
