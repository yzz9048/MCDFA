import numpy as np


def compute_event_lead_time(
    predictions,
    window_starts,
    raw_labels,
    window,
    pred_len,
):
    """Compute event-level fault lead time in timestamp units.

    A fault onset is a transition from a zero label to a non-zero label.  For
    each onset covered by at least one forecast horizon, the earliest positive
    prediction inside that horizon is used.  Each prediction can match at most
    one onset, and each real fault is counted at most once even when stride-one
    windows overlap heavily.

    Returns ``(average, detected_onsets, eligible_onsets)``.  Missed onsets are
    excluded from the average and remain visible through the two counts.
    """
    predictions = np.asarray(predictions).reshape(-1)
    window_starts = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(raw_labels).reshape(-1)

    if predictions.size != window_starts.size:
        raise ValueError(
            "predictions and window_starts must have the same length, got "
            f"{predictions.size} and {window_starts.size}."
        )
    if not 0 < pred_len < window:
        raise ValueError(
            f"pred_len must satisfy 0 < pred_len < window, got {pred_len} and {window}."
        )
    if raw_labels.size == 0:
        return float("nan"), 0, 0

    fault = raw_labels != 0
    onset_mask = np.zeros_like(fault)
    onset_mask[1:] = fault[1:] & ~fault[:-1]
    onsets = np.flatnonzero(onset_mask)

    # The last observed timestamp immediately precedes the forecast target.
    observation_ends = window_starts + window - pred_len - 1
    positive_observation_ends = observation_ends[predictions != 0]
    unused_predictions = np.ones(positive_observation_ends.size, dtype=bool)

    lead_times = []
    eligible_onsets = 0
    for onset in onsets:
        coverage = (
            (observation_ends < onset)
            & (onset <= observation_ends + pred_len)
        )
        if not np.any(coverage):
            continue

        eligible_onsets += 1
        detected = (
            unused_predictions
            & (positive_observation_ends < onset)
            & (onset <= positive_observation_ends + pred_len)
        )
        if np.any(detected):
            candidates = np.flatnonzero(detected)
            selected = candidates[
                np.argmin(positive_observation_ends[candidates])
            ]
            earliest_alarm = positive_observation_ends[selected]
            unused_predictions[selected] = False
            lead_times.append(int(onset - earliest_alarm))

    average = float(np.mean(lead_times)) if lead_times else float("nan")
    return average, len(lead_times), eligible_onsets


def compute_event_classification_metrics(
    predictions,
    window_starts,
    raw_labels,
    window,
    pred_len,
):
    """Return one-to-one alarm-event precision, recall, F1, and lead time.

    Adjacent positive windows (allowing one missing stride) form one alarm
    episode. Each episode can match at most one forecastable fault onset.
    """
    predictions = np.asarray(predictions).reshape(-1).astype(bool)
    window_starts = np.asarray(window_starts, dtype=np.int64).reshape(-1)
    raw_labels = np.asarray(raw_labels).reshape(-1)
    if predictions.size != window_starts.size:
        raise ValueError("predictions and window_starts must have the same length.")
    if not 0 < pred_len < window:
        raise ValueError("pred_len must satisfy 0 < pred_len < window.")

    observation_ends = window_starts + window - pred_len - 1
    target_ends = observation_ends + pred_len
    unique_obs = np.unique(observation_ends)
    stride = int(np.min(np.diff(unique_obs))) if unique_obs.size > 1 else 1
    merge_gap = 2 * stride

    fault = raw_labels != 0
    onset_mask = np.zeros_like(fault, dtype=bool)
    onset_mask[1:] = fault[1:] & ~fault[:-1]
    onsets = np.flatnonzero(onset_mask)
    eligible_onsets = [
        int(onset) for onset in onsets
        if np.any((observation_ends < onset) & (onset <= target_ends))
    ]

    positive_indices = np.flatnonzero(predictions)
    clusters = []
    for index in positive_indices:
        if (
            clusters
            and observation_ends[index] - observation_ends[clusters[-1][-1]] <= merge_gap
        ):
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])

    matched_onsets = set()
    matched_leads = []
    for cluster in clusters:
        candidates = []
        first_matching_index = {}
        for onset in eligible_onsets:
            matching = [
                index for index in cluster
                if observation_ends[index] < onset <= target_ends[index]
            ]
            if matching:
                candidates.append(onset)
                first_matching_index[onset] = matching[0]
        available = [onset for onset in candidates if onset not in matched_onsets]
        if available:
            matched = min(available)
            matched_onsets.add(matched)
            matched_index = first_matching_index[matched]
            matched_leads.append(int(matched - observation_ends[matched_index]))

    true_positives = len(matched_onsets)
    precision = true_positives / len(clusters) if clusters else 0.0
    recall = true_positives / len(eligible_onsets) if eligible_onsets else float("nan")
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if np.isfinite(recall) and precision + recall > 0 else 0.0
    )
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "eligible_onsets": len(eligible_onsets),
        "detected_alarm_events": len(clusters),
        "matched_alarm_events": true_positives,
        "false_alarm_events": len(clusters) - true_positives,
        "event_avg_lead_time_timestamps": (
            float(np.mean(matched_leads)) if matched_leads else float("nan")
        ),
        "event_avg_lead_time_all_eligible_onsets_timestamps": (
            float(np.sum(matched_leads) / len(eligible_onsets))
            if eligible_onsets else float("nan")
        ),
    }
