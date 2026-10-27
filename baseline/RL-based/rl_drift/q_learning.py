"""Tabular Q-learning agents (paper Algorithms 2 and 3).

Method Selector
---------------
State ``(step, previous_action, kpi_index)``.  Layer actions follow Eq. (5):
initial ``{R, F, Z}``; after R only R; after F/Z, ``{F, Z}``.  After the last
layer, terminal ``{P, D}``.  A trajectory is one complete adaptation plan.

Data Collector
--------------
State is the current data size, actions increase/decrease by the configured
interval.  One Q-table is trained per selected adaptation plan, matching the
paper's "a new agent is defined for each method".

Q update: ``Q <- (1-alpha) Q + alpha [R + gamma max Q(S', A')]``.
"""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from .adaptation import AdaptationPlan
from .config import QLearningConfig
from .model import LAYER_ACTIONS, LAYER_NAMES, TERMINAL_ACTIONS
from .utils import write_json


SelectorState = Tuple[int, str, int]
Transition = Tuple[SelectorState, str, Optional[SelectorState], Tuple[str, ...]]
CollectorState = int
CollectorTransition = Tuple[int, str, int, Tuple[str, ...]]


@dataclass
class EpsilonSchedule:
    explore_episodes: int = 2_000
    decay_every: int = 200
    decay_amount: float = 0.1
    min_epsilon: float = 0.1

    def epsilon(self, episode: int) -> float:
        """1-indexed epsilon following the paper's schedule."""
        if episode <= self.explore_episodes:
            return 1.0
        steps = 1 + (episode - self.explore_episodes - 1) // max(1, self.decay_every)
        return max(self.min_epsilon, 1.0 - self.decay_amount * steps)


class EpisodeReplay:
    def __init__(self, capacity: int):
        self.memory: Deque[Tuple[List[Transition], float]] = deque(
            maxlen=max(1, int(capacity))
        )

    def add(self, trajectory: List[Transition], reward: float) -> None:
        self.memory.append((trajectory, float(reward)))

    def sample(self, size: int, rng: random.Random) -> List[Tuple[List[Transition], float]]:
        if not self.memory:
            return []
        count = min(int(size), len(self.memory))
        return rng.sample(list(self.memory), count)

    def __len__(self) -> int:
        return len(self.memory)


class MethodSelectorAgent:
    def __init__(self, config: QLearningConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.alpha = float(config.alpha)
        self.gamma = float(config.gamma)
        self.rng = rng or random.Random()
        self.q: Dict[SelectorState, Dict[str, float]] = {}

    # ------------------------------------------------------------- policy
    @staticmethod
    def available_actions(step: int, previous_action: str) -> Tuple[str, ...]:
        if step == 1:
            return tuple(LAYER_ACTIONS)
        if step <= len(LAYER_NAMES):
            if previous_action == "R":
                return ("R",)
            return ("F", "Z")
        if step == len(LAYER_NAMES) + 1:
            return tuple(TERMINAL_ACTIONS)
        raise ValueError(f"Invalid selector step {step}.")

    def q_value(self, state: SelectorState, action: str) -> float:
        return float(self.q.get(state, {}).get(action, 0.0))

    def _argmax(self, state: SelectorState, actions: Sequence[str]) -> str:
        values = [self.q_value(state, action) for action in actions]
        best = max(values)
        candidates = [
            action for action, value in zip(actions, values)
            if value == best
        ]
        return self.rng.choice(candidates)

    def select_action(
        self,
        state: SelectorState,
        epsilon: float,
        greedy: bool = False,
    ) -> str:
        actions = self.available_actions(state[0], state[1])
        if not greedy and self.rng.random() < epsilon:
            return self.rng.choice(actions)
        return self._argmax(state, actions)

    def sample_plan(
        self,
        kpi_index: int,
        epsilon: float,
    ) -> Tuple[AdaptationPlan, List[Transition]]:
        state: SelectorState = (1, "START", int(kpi_index))
        layer_actions: List[str] = []
        transitions: List[Transition] = []
        for step in range(1, len(LAYER_NAMES) + 1):
            action = self.select_action(state, epsilon)
            layer_actions.append(action)
            if step < len(LAYER_NAMES):
                next_state: Optional[SelectorState] = (step + 1, action, kpi_index)
                next_actions = self.available_actions(step + 1, action)
            else:
                next_state = (step + 1, action, kpi_index)
                next_actions = tuple(TERMINAL_ACTIONS)
            transitions.append((state, action, next_state, next_actions))
            state = next_state
        terminal_action = self.select_action(state, epsilon)
        transitions.append((state, terminal_action, None, ()))
        plan = AdaptationPlan(tuple(layer_actions), terminal_action)
        return plan, transitions

    def greedy_plan(self, kpi_index: int) -> AdaptationPlan:
        plan, _ = self.sample_plan(kpi_index, epsilon=0.0)
        # sample_plan still uses the rng for tie-breaking; it is greedy.
        return plan

    def greedy_plan_with_trajectory(self, kpi_index: int) -> Tuple[AdaptationPlan, List[Transition]]:
        state: SelectorState = (1, "START", int(kpi_index))
        layer_actions: List[str] = []
        transitions: List[Transition] = []
        for step in range(1, len(LAYER_NAMES) + 1):
            actions = self.available_actions(step, state[1])
            action = self._argmax(state, actions)
            layer_actions.append(action)
            if step < len(LAYER_NAMES):
                next_state: Optional[SelectorState] = (step + 1, action, kpi_index)
                next_actions = self.available_actions(step + 1, action)
            else:
                next_state = (step + 1, action, kpi_index)
                next_actions = tuple(TERMINAL_ACTIONS)
            transitions.append((state, action, next_state, next_actions))
            state = next_state
        terminal_action = self._argmax(state, tuple(TERMINAL_ACTIONS))
        transitions.append((state, terminal_action, None, ()))
        return AdaptationPlan(tuple(layer_actions), terminal_action), transitions

    # ------------------------------------------------------------- update
    def update_transition(
        self,
        state: SelectorState,
        action: str,
        reward: float,
        next_state: Optional[SelectorState],
        next_actions: Sequence[str],
    ) -> None:
        next_value = 0.0
        if next_state is not None and next_actions:
            next_value = max(self.q_value(next_state, a) for a in next_actions)
        target = float(reward) + self.gamma * next_value
        current = self.q_value(state, action)
        self.q.setdefault(state, {})[action] = current + self.alpha * (target - current)

    def update_episode(self, trajectory: List[Transition], reward: float) -> None:
        for state, action, next_state, next_actions in trajectory:
            self.update_transition(state, action, reward, next_state, next_actions)

    # --------------------------------------------------------------- I/O
    def entries(self) -> List[List[object]]:
        rows: List[List[object]] = []
        for (step, previous, kpi_index), values in self.q.items():
            for action, value in values.items():
                rows.append([step, previous, kpi_index, action, float(value)])
        return rows

    def save(self, path: Path) -> None:
        write_json(path, {
            "agent": "method_selector",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q": self.entries(),
        })

    @classmethod
    def load(
        cls,
        path: Path,
        config: QLearningConfig,
        rng: Optional[random.Random] = None,
    ) -> "MethodSelectorAgent":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        agent = cls(config, rng=rng)
        for step, previous, kpi_index, action, value in data.get("q", []):
            agent.q.setdefault(
                (int(step), str(previous), int(kpi_index)), {}
            )[str(action)] = float(value)
        return agent


class DataCollectorAgent:
    def __init__(self, config: QLearningConfig, rng: Optional[random.Random] = None):
        self.config = config
        self.alpha = float(config.alpha)
        self.gamma = float(config.gamma)
        self.rng = rng or random.Random()
        self.q: Dict[Tuple[int, str], float] = {}
        self._grid: Tuple[int, ...] = ()

    def set_grid(self, grid: Sequence[int]) -> None:
        self._grid = tuple(int(value) for value in grid)

    @staticmethod
    def available_actions(size: int, grid: Sequence[int]) -> Tuple[str, ...]:
        try:
            index = list(grid).index(int(size))
        except ValueError:
            return ()
        actions: List[str] = []
        if index > 0:
            actions.append("decrease")
        if index + 1 < len(grid):
            actions.append("increase")
        return tuple(actions)

    @staticmethod
    def next_state(size: int, action: str, grid: Sequence[int]) -> Optional[int]:
        values = list(grid)
        try:
            index = values.index(int(size))
        except ValueError:
            return None
        if action == "increase" and index + 1 < len(values):
            return int(values[index + 1])
        if action == "decrease" and index > 0:
            return int(values[index - 1])
        return None

    def q_value(self, size: int, action: str) -> float:
        return float(self.q.get((int(size), action), 0.0))

    def _argmax(self, size: int, actions: Sequence[str]) -> str:
        values = [self.q_value(size, action) for action in actions]
        best = max(values)
        candidates = [a for a, v in zip(actions, values) if v == best]
        return self.rng.choice(candidates)

    def select_action(
        self,
        size: int,
        grid: Sequence[int],
        epsilon: float,
        greedy: bool = False,
    ) -> Optional[str]:
        actions = self.available_actions(size, grid)
        if not actions:
            return None
        if not greedy and self.rng.random() < epsilon:
            return self.rng.choice(actions)
        return self._argmax(size, actions)

    def update_transition(
        self,
        size: int,
        action: str,
        reward: float,
        next_size: Optional[int],
        grid: Sequence[int],
    ) -> None:
        next_value = 0.0
        if next_size is not None:
            next_actions = self.available_actions(next_size, grid)
            if next_actions:
                next_value = max(self.q_value(next_size, a) for a in next_actions)
        target = float(reward) + self.gamma * next_value
        key = (int(size), action)
        current = self.q_value(int(size), action)
        self.q[key] = current + self.alpha * (target - current)

    def entries(self) -> List[List[object]]:
        return [[int(size), action, float(value)] for (size, action), value in self.q.items()]

    def save(self, path: Path) -> None:
        write_json(path, {
            "agent": "data_collector",
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q": self.entries(),
        })

    @classmethod
    def load(
        cls,
        path: Path,
        config: QLearningConfig,
        rng: Optional[random.Random] = None,
    ) -> "DataCollectorAgent":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        agent = cls(config, rng=rng)
        for size, action, value in data.get("q", []):
            agent.q[(int(size), str(action))] = float(value)
        return agent
