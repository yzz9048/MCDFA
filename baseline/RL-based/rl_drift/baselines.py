"""Validation-only baseline comparisons for the paper's evaluation table.

The paper compares against random method selection, always-retrain, three
transfer-learning variants, incremental/decremental data collection, and a
human expert.  A human expert is not available here, so ``oracle_cached`` is
reported explicitly as an oracle proxy (best cached adaptation for the same
event), never as a human result.

All measurements use the validation drift stream and validation accuracy
holdout.  Test data is not used.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .adaptation import AdaptationEnvironment, AdaptationPlan
from .config import Config, KPISet
from .q_learning import DataCollectorAgent
from .utils import write_json


def _reward(config: Config, accuracy: float, seconds: float, resource: float, kpi: KPISet) -> float:
    required = max(float(kpi.accuracy), 1e-9)
    cfg = config.reward
    return float(
        cfg.k1 * (accuracy - required) / required
        - cfg.k2 * seconds / max(float(kpi.time), 1e-9)
        - cfg.k3 * resource / max(float(kpi.resource), 1e-9)
    )


def _finite_mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def _choose_size(
    config: Config,
    collector: DataCollectorAgent,
    grid: Sequence[int],
) -> int:
    if len(grid) == 1:
        return int(grid[0])
    target = int(config.data_collector.deployment_initial_size)
    size = min(grid, key=lambda value: abs(int(value) - target))
    seen = {int(size)}
    for _ in range(int(config.qlearning.collector_episode_length)):
        actions = collector.available_actions(size, grid)
        if not actions:
            break
        values = [collector.q_value(size, action) for action in actions]
        if max(values) == 0.0 and min(values) == 0.0:
            break
        action = collector.select_action(size, grid, epsilon=0.0, greedy=True)
        if action is None:
            break
        next_size = collector.next_state(size, action, grid)
        if next_size is None or next_size in seen:
            break
        seen.add(int(next_size))
        size = int(next_size)
    return int(size)


def compute_baselines(
    config: Config,
    env: AdaptationEnvironment,
    selector,
    collector_lookup: Callable[[AdaptationPlan, int], DataCollectorAgent],
    kpi: KPISet,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    kpi_index = config.kpi_sets.index(kpi)
    greedy_plan = selector.greedy_plan_with_trajectory(kpi_index)[0]
    collector = collector_lookup(greedy_plan, kpi_index)
    rng = random.Random(config.seed + 991)
    all_plans = AdaptationPlan.all_plans()
    initial = env.initial_holdout_metrics()
    no_adapt_reward = _reward(config, initial["accuracy"], 0.0, 0.0, kpi)

    fixed_methods: Dict[str, AdaptationPlan] = {
        "always_retrain": AdaptationPlan(("R", "R", "R", "R"), "D"),
        "tl1_lower_layers": AdaptationPlan(("F", "F", "Z", "Z"), "D"),
        "tl2_end_layers": AdaptationPlan(("Z", "Z", "F", "F"), "D"),
        "tl3_all_layers": AdaptationPlan(("F", "F", "F", "F"), "D"),
    }

    rows: List[Dict[str, object]] = []
    per_event_oracle: Dict[int, float] = {}

    for event in env.events:
        grid = env.data_grid(event)
        if not grid:
            continue
        max_size = int(grid[-1])

        rows.append({
            "event_id": event.event_id,
            "method": "no_adaptation",
            "plan": "",
            "data_size": 0,
            "accuracy": initial["accuracy"],
            "train_seconds": 0.0,
            "resource_mb": 0.0,
            "reward": no_adapt_reward,
        })

        for name, plan in fixed_methods.items():
            metrics, reward = env.evaluate_and_reward(plan, event, max_size, kpi)
            rows.append(_row(event.event_id, name, metrics, reward))

        random_plan = rng.choice(all_plans)
        metrics, reward = env.evaluate_and_reward(random_plan, event, max_size, kpi)
        rows.append(_row(event.event_id, "random_method", metrics, reward))

        # RL-greedy method + data collector.
        rl_size = _choose_size(config, collector, grid)
        metrics, reward = env.evaluate_and_reward(greedy_plan, event, rl_size, kpi)
        rows.append(_row(event.event_id, "rl_method_and_collector", metrics, reward))

        # Incremental / decremental data collection for the greedy method.
        inc_seconds = inc_resource = 0.0
        inc_metrics = None
        for size in grid:
            metrics = env.evaluate(greedy_plan, event, size)
            inc_seconds += metrics.train_seconds
            inc_resource += metrics.resource_mb
            inc_metrics = metrics
            if metrics.accuracy >= kpi.accuracy:
                break
        if inc_metrics is not None:
            inc_reward = _reward(
                config, inc_metrics.accuracy, inc_seconds, inc_resource, kpi
            )
            rows.append({
                "event_id": event.event_id,
                "method": "incremental_data",
                "plan": greedy_plan.name,
                "data_size": inc_metrics.data_size,
                "accuracy": inc_metrics.accuracy,
                "train_seconds": inc_seconds,
                "resource_mb": inc_resource,
                "reward": inc_reward,
            })

        dec_seconds = dec_resource = 0.0
        dec_metrics = None
        for size in reversed(grid):
            metrics = env.evaluate(greedy_plan, event, size)
            dec_seconds += metrics.train_seconds
            dec_resource += metrics.resource_mb
            if metrics.accuracy >= kpi.accuracy:
                dec_metrics = metrics
                break
        if dec_metrics is None:
            # Requirement was never met; report the largest-data attempt.
            dec_metrics = env.evaluate(greedy_plan, event, max_size)
        dec_reward = _reward(
            config, dec_metrics.accuracy, dec_seconds, dec_resource, kpi
        )
        rows.append({
            "event_id": event.event_id,
            "method": "decremental_data",
            "plan": greedy_plan.name,
            "data_size": dec_metrics.data_size,
            "accuracy": dec_metrics.accuracy,
            "train_seconds": dec_seconds,
            "resource_mb": dec_resource,
            "reward": dec_reward,
        })

        # Oracle proxy over every adaptation cached for this event.
        cached_rewards = [
            env.reward(metrics, kpi)
            for (event_id, _, _), metrics in env.cache.items()
            if event_id == event.event_id
        ]
        oracle = max(cached_rewards) if cached_rewards else no_adapt_reward
        per_event_oracle[event.event_id] = oracle
        rows.append({
            "event_id": event.event_id,
            "method": "oracle_cached",
            "plan": "",
            "data_size": 0,
            "accuracy": float("nan"),
            "train_seconds": 0.0,
            "resource_mb": 0.0,
            "reward": oracle,
        })

    for row in rows:
        oracle = per_event_oracle.get(int(row["event_id"]))
        row["regret_vs_oracle_cached"] = (
            float(oracle - row["reward"]) if oracle is not None else float("nan")
        )

    methods = sorted({str(row["method"]) for row in rows})
    aggregate: List[Dict[str, object]] = []
    for method in methods:
        subset = [row for row in rows if row["method"] == method]
        aggregate.append({
            "method": method,
            "events": len(subset),
            "mean_accuracy": _finite_mean([
                float(row["accuracy"]) for row in subset
            ]),
            "mean_train_seconds": _finite_mean([
                float(row["train_seconds"]) for row in subset
            ]),
            "mean_resource_mb": _finite_mean([
                float(row["resource_mb"]) for row in subset
            ]),
            "mean_reward": _finite_mean([
                float(row["reward"]) for row in subset
            ]),
            "mean_regret_vs_oracle_cached": _finite_mean([
                float(row["regret_vs_oracle_cached"]) for row in subset
            ]),
        })
    return rows, aggregate


def _row(event_id: int, method: str, metrics, reward: float) -> Dict[str, object]:
    return {
        "event_id": event_id,
        "method": method,
        "plan": metrics.plan,
        "data_size": metrics.data_size,
        "accuracy": metrics.accuracy,
        "train_seconds": metrics.train_seconds,
        "resource_mb": metrics.resource_mb,
        "reward": reward,
    }


def save_baselines(
    path_dir: Path,
    rows: List[Dict[str, object]],
    aggregate: List[Dict[str, object]],
) -> None:
    path_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(path_dir / "baselines_per_event.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    write_json(path_dir / "baselines_aggregate.json", aggregate)
