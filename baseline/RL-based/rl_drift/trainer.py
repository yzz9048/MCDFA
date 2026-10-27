"""Predictor training and offline Q-learning (paper Algorithms 2 and 3)."""

from __future__ import annotations

import csv
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from .adaptation import (
    AdaptationEnvironment, AdaptationPlan, classification_metrics,
)
from .config import Config, PredictorConfig
from .data import PaperDataModule
from .drift_detector import DetectionResult, DriftEvent, find_natural_drift_events
from .model import FaultCNN
from .q_learning import (
    DataCollectorAgent, EpisodeReplay, EpsilonSchedule, MethodSelectorAgent,
)
from .utils import resolve_device, write_json


def _slug(plan: AdaptationPlan) -> str:
    return plan.name.replace(",", "-").replace("|", "_")


def class_weights(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    counts = torch.bincount(labels.long().cpu(), minlength=num_classes).float()
    total = counts.sum()
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    return weights


def save_predictor(path: Path, model: FaultCNN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "input_dim": int(model.input_dim),
        "input_window": int(model.input_window),
        "predictor_config": asdict(model.config),
    }, path)


def load_predictor(path: Path, device: torch.device) -> FaultCNN:
    state = torch.load(path, map_location=device)
    cfg = PredictorConfig(**state["predictor_config"])
    model = FaultCNN(
        int(state["input_dim"]), int(state["input_window"]), config=cfg
    )
    model.load_state_dict(state["state_dict"])
    model.to(device)
    model.eval()
    return model


def train_initial_predictor(
    data: PaperDataModule,
    config: Config,
    device: torch.device,
    log_path: Optional[Path] = None,
) -> Tuple[FaultCNN, List[Dict[str, object]]]:
    """Train the paper's CNN on the fit slice, early-stop on its tail.

    No validation-stream or test data is touched.
    """
    random.seed(config.seed)
    np.random.seed(config.seed % (2 ** 32))
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    n_features = len(data.selected_indices)
    p = config.predictor
    model = FaultCNN(
        n_features, p.input_window, config=p, num_classes=2
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=p.lr, weight_decay=p.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(
        torch.from_numpy(data.fit_windows.labels)
    ).to(device))
    x = torch.from_numpy(data.fit_windows.features).to(device)
    y = torch.from_numpy(data.fit_windows.labels).long().to(device)
    x_val = torch.from_numpy(data.predictor_val_windows.features).to(device)
    y_val = torch.from_numpy(data.predictor_val_windows.labels).long().to(device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed)
    batch_size = max(1, int(p.batch_size))

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_accuracy = -1.0
    stale = 0
    rows: List[Dict[str, object]] = []
    for epoch in range(1, int(p.epochs) + 1):
        model.train()
        order = torch.randperm(int(y.shape[0]), generator=generator)
        losses: List[float] = []
        for start in range(0, int(y.shape[0]), batch_size):
            index = order[start:start + batch_size].to(device)
            logits = model(x[index])
            loss = loss_fn(logits, y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([
                model(x_val[start:start + 512])
                for start in range(0, x_val.shape[0], 512)
            ], dim=0)
            scores = classification_metrics(torch.softmax(val_logits, dim=1), y_val)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "predictor_val_accuracy": scores["accuracy"],
            "predictor_val_balanced_accuracy": scores["balanced_accuracy"],
        }
        rows.append(row)
        if scores["accuracy"] > best_accuracy + 1e-9:
            best_accuracy = scores["accuracy"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= p.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["epoch"])
            writer.writeheader()
            writer.writerows(rows)
    return model, rows


class RLTrainer:
    def __init__(
        self,
        config: Config,
        data: PaperDataModule,
        initial_model: FaultCNN,
        device: Optional[torch.device] = None,
    ):
        self.config = config
        self.data = data
        self.device = device or resolve_device(config.device)
        self.initial_model = initial_model.to(self.device)
        self.results_dir = config.results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(config.seed)

        detection = self.predict_stream()
        self.detection: DetectionResult = find_natural_drift_events(
            detection,
            data.stream_windows.labels,
            config,
            timestamps=data.stream_windows.timestamps,
        )
        self.events: List[DriftEvent] = self.detection.events
        self.env = AdaptationEnvironment(
            config, data, self.events, self.initial_model, self.device
        )
        self.selector = MethodSelectorAgent(config.qlearning, rng=self.rng)
        self.collectors: Dict[Tuple[str, int], DataCollectorAgent] = {}
        self.selector_log: List[Dict[str, object]] = []
        self.collector_logs: Dict[Tuple[str, int], List[Dict[str, object]]] = {}
        self.epsilon_schedule = EpsilonSchedule(
            explore_episodes=config.qlearning.explore_episodes,
            decay_every=config.qlearning.decay_every,
            decay_amount=config.qlearning.decay_amount,
            min_epsilon=config.qlearning.min_epsilon,
        )

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def predict_stream(self) -> np.ndarray:
        x = torch.from_numpy(self.data.stream_windows.features).to(self.device)
        chunks = [
            self.initial_model(x[start:start + 512]).argmax(dim=1).cpu().numpy()
            for start in range(0, x.shape[0], 512)
        ]
        return np.concatenate(chunks) if chunks else np.empty((0,), dtype=np.int64)

    def event_summary(self) -> List[Dict[str, object]]:
        return [event.as_dict() for event in self.events]

    # ---------------------------------------------------------- offline RL
    def _log_selector(self, row: Dict[str, object]) -> None:
        self.selector_log.append(row)

    def train_method_selector(self) -> MethodSelectorAgent:
        cfg = self.config.qlearning
        replay = EpisodeReplay(cfg.replay_capacity)
        kpi_sets = self.config.kpi_sets
        log_every = max(1, int(cfg.selector_episodes) // 20)
        moving: List[float] = []
        for episode in range(1, int(cfg.selector_episodes) + 1):
            event = self.rng.choice(self.events)
            kpi_index = self.rng.randrange(len(kpi_sets))
            kpi = kpi_sets[kpi_index]
            epsilon = self.epsilon_schedule.epsilon(episode)
            plan, trajectory = self.selector.sample_plan(kpi_index, epsilon)
            data_size = self.env.selector_data_size(event)
            metrics, reward = self.env.evaluate_and_reward(
                plan, event, data_size, kpi
            )
            replay.add(trajectory, reward)
            for stored_trajectory, stored_reward in replay.sample(
                cfg.replay_size, self.rng
            ):
                self.selector.update_episode(stored_trajectory, stored_reward)
            moving.append(reward)
            if len(moving) > 100:
                moving.pop(0)
            row = {
                "episode": episode,
                "epsilon": epsilon,
                "event_id": event.event_id,
                "kpi": kpi.name,
                "kpi_index": kpi_index,
                "plan": plan.name,
                "data_size": metrics.data_size,
                "requested_data_size": metrics.requested_data_size,
                "accuracy": metrics.accuracy,
                "balanced_accuracy": metrics.balanced_accuracy,
                "train_seconds": metrics.train_seconds,
                "resource_mb": metrics.resource_mb,
                "reward": reward,
                "cached": metrics.cached,
                "moving_reward_100": float(np.mean(moving)),
            }
            self._log_selector(row)
            if episode % log_every == 0 or episode == 1:
                print(
                    f"[selector] episode {episode}/{cfg.selector_episodes} "
                    f"eps={epsilon:.2f} reward={reward:+.4f} "
                    f"moving={row['moving_reward_100']:+.4f} "
                    f"plan={plan.name} size={metrics.data_size} "
                    f"acc={metrics.accuracy:.4f}"
                )
        self.selector.save(self.results_dir / "selector_q.json")
        self._write_csv(
            self.results_dir / "selector_training.csv", self.selector_log
        )
        return self.selector

    def _write_collector_csv(
        self, plan: AdaptationPlan, kpi_index: int,
        rows: List[Dict[str, object]],
    ) -> None:
        path = (
            self.results_dir / "collectors"
            / f"{_slug(plan)}_kpi{kpi_index}_training.csv"
        )
        self._write_csv(path, rows)

    def train_data_collector(
        self,
        plan: AdaptationPlan,
        kpi_index: int,
    ) -> DataCollectorAgent:
        cfg = self.config.qlearning
        kpi = self.config.kpi_sets[kpi_index]
        agent = DataCollectorAgent(cfg, rng=self.rng)
        rows: List[Dict[str, object]] = []
        log_every = max(1, int(cfg.collector_episodes) // 20)
        moving: List[float] = []
        for episode in range(1, int(cfg.collector_episodes) + 1):
            event = self.rng.choice(self.events)
            grid = self.env.data_grid(event)
            if len(grid) < 2:
                continue
            size = self.rng.choice(grid)
            epsilon = self.epsilon_schedule.epsilon(episode)
            for _ in range(int(cfg.collector_episode_length)):
                action = agent.select_action(size, grid, epsilon)
                if action is None:
                    break
                next_size = agent.next_state(size, action, grid)
                if next_size is None:
                    break
                accuracy = self.env.accuracy_only(plan, event, next_size)
                reward = float(accuracy) - float(kpi.accuracy)
                agent.update_transition(size, action, reward, next_size, grid)
                moving.append(reward)
                if len(moving) > 100:
                    moving.pop(0)
                rows.append({
                    "episode": episode,
                    "epsilon": epsilon,
                    "event_id": event.event_id,
                    "kpi": kpi.name,
                    "plan": plan.name,
                    "action": action,
                    "size": size,
                    "next_size": next_size,
                    "accuracy": accuracy,
                    "reward": reward,
                    "moving_reward_100": float(np.mean(moving)),
                })
                size = next_size
                if reward >= 0.0:
                    break
            if episode % log_every == 0 or episode == 1:
                print(
                    f"[collector] {plan.name} kpi={kpi.name} "
                    f"episode {episode}/{cfg.collector_episodes} "
                    f"eps={epsilon:.2f} q_states={len(agent.q)}"
                )
        self.collectors[(plan.name, kpi_index)] = agent
        self.collector_logs[(plan.name, kpi_index)] = rows
        path = (
            self.results_dir / "collectors"
            / f"{_slug(plan)}_kpi{kpi_index}.json"
        )
        agent.save(path)
        self._write_collector_csv(plan, kpi_index, rows)
        return agent

    @staticmethod
    def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        # Keep the union of keys in insertion order across rows.
        fieldnames: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})

    def train(self) -> Dict[str, object]:
        print(
            f"[rl] dataset={self.config.dataset} events={len(self.events)} "
            f"stream_windows={len(self.data.stream_windows)} "
            f"holdout_windows={len(self.data.accuracy_windows)} "
            f"device={self.device}"
        )
        # Regenerated artifacts: do not let collector Q-tables from an older
        # run with a different greedy-plan set leak into this run.  Only
        # generated collector files are removed.
        collectors_dir = self.results_dir / "collectors"
        collectors_dir.mkdir(parents=True, exist_ok=True)
        for pattern in ("*_kpi*.json", "*_kpi*_training.csv"):
            for stale in collectors_dir.glob(pattern):
                if stale.is_file():
                    stale.unlink()
        for stale in (
            self.results_dir / "selector_q.json",
            self.results_dir / "adaptation_cache.json",
            self.results_dir / "adaptation_cache.csv",
            self.results_dir / "selector_training.csv",
        ):
            if stale.exists():
                stale.unlink()

        # Make train() idempotent if called more than once on one trainer.
        self.rng = random.Random(self.config.seed)
        self.selector = MethodSelectorAgent(
            self.config.qlearning, rng=self.rng
        )
        self.collectors.clear()
        self.selector_log.clear()
        self.collector_logs.clear()
        self.env.cache.clear()

        self._write_csv(self.results_dir / "drift_events.csv", self.event_summary())
        self.train_method_selector()

        greedy: Dict[int, AdaptationPlan] = {
            index: self.selector.greedy_plan_with_trajectory(index)[0]
            for index in range(len(self.config.kpi_sets))
        }
        for index, plan in greedy.items():
            print(
                f"[selector] greedy plan for {self.config.kpi_sets[index].name}: "
                f"{plan.name}"
            )
        unique_plans = {plan.name: plan for plan in greedy.values()}
        for plan in unique_plans.values():
            for kpi_index in range(len(self.config.kpi_sets)):
                self.train_data_collector(plan, kpi_index)

        self.env.save_cache(self.results_dir / "adaptation_cache.json")
        summary = {
            "events": self.event_summary(),
            "initial_holdout": self.env.initial_holdout_metrics(),
            "greedy_plans": {
                self.config.kpi_sets[index].name: plan.name
                for index, plan in greedy.items()
            },
            "selector_cache_entries": len(self.env.cache),
            "selector_training_rows": len(self.selector_log),
            "collector_training_rows": {
                f"{plan}|{self.config.kpi_sets[kpi].name}": len(rows)
                for (plan, kpi), rows in self.collector_logs.items()
            },
            "data": self.data.summary(),
            "config": self.config.to_dict(),
        }
        write_json(self.results_dir / "training_summary.json", summary)
        return summary

    # ------------------------------------------------------------- loading
    def load_trained_agents(self) -> None:
        selector_path = self.results_dir / "selector_q.json"
        if not selector_path.exists():
            raise FileNotFoundError(f"Missing trained selector: {selector_path}")
        self.selector = MethodSelectorAgent.load(
            selector_path, self.config.qlearning, rng=self.rng
        )
        cache_path = self.results_dir / "adaptation_cache.json"
        if not cache_path.exists():
            # Backward compatibility with an earlier generated filename.
            cache_path = self.results_dir / "adaptation_cache.csv"
        if cache_path.exists():
            self.env.load_cache(cache_path)
        self.collectors.clear()
        collectors_dir = self.results_dir / "collectors"
        if collectors_dir.exists():
            for path in sorted(collectors_dir.glob("*_kpi*.json")):
                stem = path.stem
                if "_kpi" not in stem:
                    continue
                plan_stem, kpi_stem = stem.rsplit("_kpi", 1)
                try:
                    kpi_index = int(kpi_stem)
                except ValueError:
                    continue
                plan_name = plan_stem.replace("-", ",").replace("_", "|")
                try:
                    plan = AdaptationPlan.from_name(plan_name)
                except ValueError:
                    continue
                self.collectors[(plan.name, kpi_index)] = DataCollectorAgent.load(
                    path, self.config.qlearning, rng=self.rng
                )

    def collector_for(
        self, plan: AdaptationPlan, kpi_index: int
    ) -> DataCollectorAgent:
        key = (plan.name, kpi_index)
        if key in self.collectors:
            return self.collectors[key]
        # Untrained collector: no actions, deployment falls back to the
        # configured initial size.  This is explicit rather than silent.
        return DataCollectorAgent(self.config.qlearning, rng=self.rng)
