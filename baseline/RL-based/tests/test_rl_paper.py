"""Fast tests for the RL-based paper pipeline on a synthetic dataset."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rl_drift.adaptation import (
    AdaptationEnvironment, AdaptationPlan, classification_metrics,
)
from rl_drift.baselines import compute_baselines
from rl_drift.config import (
    AdaptationConfig, Config, DataCollectorConfig, DetectorConfig, KPISet,
    PredictorConfig, QLearningConfig, RewardConfig, SplitConfig, validate_config,
)
from rl_drift.data import PaperDataModule
from rl_drift.drift_detector import (
    DriftDetector, DriftEvent, find_natural_drift_events, rolling_accuracy_curve,
)
from rl_drift.model import FaultCNN, ModelBank
from rl_drift.protocol import OnlineDriftHandler, evaluate_frozen_model
from rl_drift.q_learning import (
    DataCollectorAgent, EpsilonSchedule, MethodSelectorAgent,
)
from rl_drift.trainer import RLTrainer, train_initial_predictor

torch.set_num_threads(1)


def _write_dataset(path: Path, n: int = 1600, features: int = 8) -> None:
    rng = np.random.default_rng(7)
    t = np.arange(n)
    values = rng.normal(0.0, 1.0, size=(n, features))
    values[:, 0] += t / n * 2.0
    values[t >= int(0.55 * n)] += 1.2
    values[t >= int(0.8 * n)] += 1.5
    labels = np.zeros(n, dtype=int)
    for onset in range(100, n, 150):
        labels[onset:onset + 18] = 1
    frame = pd.DataFrame(values, columns=[f"f{i}" for i in range(features)])
    frame.insert(0, "timestamp", [f"t{i:05d}" for i in range(n)])
    frame.insert(1, "rps", rng.integers(100, 300, size=n))
    frame["label"] = labels
    frame.to_csv(path, index=False)


def _config(tmp_path: Path, dataset: str = "synthetic") -> Config:
    data_path = tmp_path / "synthetic.csv"
    if not data_path.exists():
        _write_dataset(data_path)
    cfg = Config(
        seed=11,
        device="cpu",
        output_root=str(tmp_path / "results"),
        dataset=dataset,
        data_root=str(data_path),
        active_kpi="test",
        kpi_sets=[KPISet("test", time=10.0, resource=10.0, accuracy=0.6)],
        predictor=PredictorConfig(
            input_window=8, forecast_horizon=4, rfe_features=5,
            conv1_out=8, conv2_out=16, dense1=32, dropout=0.1,
            epochs=2, batch_size=16, lr=1e-3, val_fraction=0.25, patience=2,
        ),
        split=SplitConfig(train=0.6, validation=0.2, test=0.2,
                          rl_stream_fraction=0.7),
        detector=DetectorConfig(
            rolling_window=8, accuracy_threshold=0.9, cooldown=5,
            min_events=2, max_events=3, min_post_windows=20,
            fallback_to_accuracy_minima=True,
        ),
        reward=RewardConfig(k1=0.8, k2=0.1, k3=0.1),
        qlearning=QLearningConfig(
            alpha=0.2, gamma=0.9, replay_size=2, replay_capacity=10,
            selector_episodes=6, collector_episodes=6, explore_episodes=2,
            decay_every=2, decay_amount=0.2, min_epsilon=0.1,
            collector_episode_length=2, selector_data_size="max",
        ),
        data_collector=DataCollectorConfig(
            D0=20, Dr=80, interval=20, deployment_initial_size=40,
        ),
        adaptation=AdaptationConfig(
            epochs=1, batch_size=8, lr=1e-3, cache=True, max_data_size=80,
            resource_metric="footprint",
        ),
    )
    validate_config(cfg)
    return cfg


@pytest.fixture()
def data_bundle(tmp_path):
    cfg = _config(tmp_path)
    data = PaperDataModule(cfg)
    return cfg, data, tmp_path


def test_chronological_split_windows_and_no_overlap(data_bundle):
    cfg, data, _ = data_bundle
    assert data.fit_span[1] == data.predictor_val_span[0]
    assert data.predictor_val_span[1] == data.validation_span[0]
    assert data.validation_span[1] == data.test_span[0]
    assert data.test_windows.starts.min() >= data.test_span[0]
    assert data.test_windows.starts.max() + cfg.predictor.input_window + \
        cfg.predictor.forecast_horizon <= data.test_span[1]

    # Every label is the OR over the next horizon; no boundary crossing.
    for windows, span in (
        (data.fit_windows, data.fit_span),
        (data.stream_windows, data.stream_span),
        (data.accuracy_windows, data.accuracy_span),
        (data.test_windows, data.test_span),
    ):
        assert windows.starts.min() >= span[0]
        assert windows.starts.max() + cfg.predictor.input_window <= span[1]
        sample = 0
        start = int(windows.starts[sample])
        expected = int(data.raw_labels[
            start + cfg.predictor.input_window:
            start + cfg.predictor.input_window + cfg.predictor.forecast_horizon
        ].any())
        assert int(windows.labels[sample]) == expected
    data.assert_no_test_leakage()


def test_rfe_and_scaler_are_fit_on_fit_slice_only(data_bundle):
    cfg, data, _ = data_bundle
    fit_start, fit_end = data.fit_span
    expected_mean = data.raw_features[fit_start:fit_end].mean(axis=0)
    assert np.allclose(data.scaler.mean_, expected_mean)
    assert len(data.selected_indices) == cfg.predictor.rfe_features
    assert len(set(data.selected_indices.tolist())) == len(data.selected_indices)


def test_cnn_forward_layer_masks_and_model_bank():
    cfg = PredictorConfig(
        input_window=8, forecast_horizon=4, rfe_features=5,
        conv1_out=8, conv2_out=16, dense1=32,
    )
    model = FaultCNN(5, 8, config=cfg)
    x = torch.randn(4, 8, 5)
    assert model(x).shape == (4, 2)

    before = {name: [p.detach().clone() for p in model.layer_parameters(name)]
              for name in ("conv1", "conv2", "dense1", "dense2")}
    model.set_action_mask({"conv1": "F", "conv2": "F", "dense1": "Z", "dense2": "Z"})
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    for index, param in enumerate(model.layer_parameters("dense1")):
        assert torch.equal(param.detach(), before["dense1"][index])
    assert any(
        not torch.equal(param.detach(), before["conv1"][index])
        for index, param in enumerate(model.layer_parameters("conv1"))
    )

    reset_before = model.conv1.weight.detach().clone()
    model.reset_all_layers()
    assert not torch.equal(model.conv1.weight.detach(), reset_before)

    bank = ModelBank(model)
    new_model = model.clone()
    bank.replace(new_model, preserve_old=True)
    assert len(bank.members()) == 2
    probs = bank.predict_proba(x)
    direct = 0.5 * (model.predict_proba(x) + new_model.predict_proba(x))
    assert torch.allclose(probs, direct)
    bank.replace(model.clone(), preserve_old=False)
    assert len(bank.members()) == 1


def test_classification_metrics_are_correct():
    labels = torch.tensor([0, 0, 1, 1])
    probs = torch.tensor([
        [0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6],
    ])
    metrics = classification_metrics(probs, labels)
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5


def test_drift_detector_fixed_threshold_and_cooldown():
    detector = DriftDetector(DetectorConfig(
        rolling_window=3, accuracy_threshold=0.5, cooldown=2,
    ))
    assert detector.update(True, 0) is None
    assert detector.update(False, 1) is None
    assert detector.update(False, 2) is not None
    assert detector.update(False, 3) is None
    assert detector.update(False, 4) is not None
    curve = rolling_accuracy_curve(np.array([1, 0, 1, 0]), 2)
    assert np.isnan(curve[0])
    assert np.allclose(curve[1:], [0.5, 0.5, 0.5])


def test_natural_drift_events_require_post_data(tmp_path):
    cfg = _config(tmp_path)
    cfg.detector.min_post_windows = 20
    n = 120
    labels = np.zeros(n, dtype=int)
    labels[50:60] = 1
    predictions = np.zeros(n, dtype=int)
    result = find_natural_drift_events(predictions, labels, cfg)
    assert result.events
    for event in result.events:
        assert event.available_windows >= 20
        assert event.anchor + 1 + event.available_windows == n


def test_qlearning_eq5_actions_and_update():
    cfg = QLearningConfig(alpha=0.5, gamma=0.5, replay_size=1)
    agent = MethodSelectorAgent(cfg, rng=random.Random(0))
    assert agent.available_actions(1, "START") == ("R", "F", "Z")
    assert agent.available_actions(2, "R") == ("R",)
    assert agent.available_actions(2, "F") == ("F", "Z")
    assert agent.available_actions(2, "Z") == ("F", "Z")
    assert agent.available_actions(5, "Z") == ("P", "D")

    state = (1, "START", 0)
    next_state = (2, "F", 0)
    agent.q[state] = {"R": 1.0, "F": 2.0, "Z": 0.0}
    agent.q[next_state] = {"F": 1.0, "Z": 0.0}
    agent.update_transition(state, "F", 1.0, next_state, ("F", "Z"))
    # target = 1 + 0.5 * 1 = 1.5; new = 2 + 0.5 * (1.5 - 2) = 1.75
    assert agent.q[state]["F"] == pytest.approx(1.75)

    plans = AdaptationPlan.all_plans()
    assert len(plans) == 34
    assert all(plan.name for plan in plans)


def test_epsilon_schedule_matches_paper_shape():
    schedule = EpsilonSchedule(
        explore_episodes=2000, decay_every=200, decay_amount=0.1,
        min_epsilon=0.1,
    )
    assert schedule.epsilon(1) == 1.0
    assert schedule.epsilon(2000) == 1.0
    assert schedule.epsilon(2001) == pytest.approx(0.9)
    assert schedule.epsilon(2201) == pytest.approx(0.8)
    assert schedule.epsilon(12000) == pytest.approx(0.1)


def test_data_collector_actions_and_updates():
    cfg = QLearningConfig(alpha=0.5, gamma=0.5)
    agent = DataCollectorAgent(cfg, rng=random.Random(0))
    grid = [10, 20, 30]
    assert agent.available_actions(10, grid) == ("increase",)
    assert agent.available_actions(20, grid) == ("decrease", "increase")
    assert agent.available_actions(30, grid) == ("decrease",)
    assert agent.next_state(20, "increase", grid) == 30
    assert agent.next_state(10, "decrease", grid) is None
    agent.update_transition(20, "increase", 0.3, 30, grid)
    assert agent.q_value(20, "increase") == pytest.approx(0.15)


def test_adaptation_environment_cache_footprint_and_reward(data_bundle):
    cfg, data, tmp_path = data_bundle
    model = FaultCNN(5, 8, config=cfg.predictor)
    event = DriftEvent(
        event_id=0, anchor=0, available_windows=len(data.stream_windows) - 1,
        reason="test", rolling_accuracy=0.5,
    )
    env = AdaptationEnvironment(cfg, data, [event], model, torch.device("cpu"))
    plan = AdaptationPlan(("F", "F", "Z", "Z"), "D")
    first = env.evaluate(plan, event, 20)
    second = env.evaluate(plan, event, 20)
    assert first.cached is False
    assert second.cached is True
    assert first.data_size == 20
    assert first.resource_mb > 0
    assert first.footprint_mb > 0
    assert np.isfinite(first.accuracy)
    assert np.isfinite(env.reward(first, cfg.kpi()))

    cache_path = tmp_path / "adaptation_cache.json"
    env.save_cache(cache_path)
    reloaded = AdaptationEnvironment(cfg, data, [event], model, torch.device("cpu"))
    assert reloaded.load_cache(cache_path) == 1
    third = reloaded.evaluate(plan, event, 20)
    assert third.cached is True
    assert third.accuracy == first.accuracy
    assert third.train_seconds == first.train_seconds


def test_online_adaptation_ignores_test_label_changes(data_bundle):
    cfg, data, _ = data_bundle
    model, _ = train_initial_predictor(data, cfg, torch.device("cpu"))
    trainer = RLTrainer(cfg, data, model, torch.device("cpu"))
    trainer.train()
    trainer.load_trained_agents()
    handler = OnlineDriftHandler(
        cfg, data, model, trainer.selector, trainer.collector_for,
        trainer.env, torch.device("cpu"), cfg.kpi(),
    )
    first = handler.run()

    original = data.test_windows.labels.copy()
    try:
        data.test_windows.labels[:] = 1 - original
        handler2 = OnlineDriftHandler(
            cfg, data, model, trainer.selector, trainer.collector_for,
            trainer.env, torch.device("cpu"), cfg.kpi(),
        )
        second = handler2.run()
    finally:
        data.test_windows.labels[:] = original

    assert second.alerts == first.alerts
    assert second.consumed_windows == first.consumed_windows
    assert second.final_holdout == first.final_holdout
    assert [row["plan"] for row in second.adaptations] == \
        [row["plan"] for row in first.adaptations]


class _TestWindowGuard:
    """Expose only the metadata that reporting is allowed to read."""

    def __init__(self, windows):
        self._windows = windows

    def __len__(self):
        return len(self._windows)

    @property
    def labels(self):
        return self._windows.labels

    @property
    def starts(self):
        return self._windows.starts

    @property
    def timestamps(self):
        return self._windows.timestamps

    @property
    def features(self):
        raise AssertionError("test features were read during RL training/online.")


def test_training_and_online_do_not_read_test_features(data_bundle):
    cfg, data, _ = data_bundle
    model, _ = train_initial_predictor(data, cfg, torch.device("cpu"))
    original = data.test_windows
    data.test_windows = _TestWindowGuard(original)
    try:
        trainer = RLTrainer(cfg, data, model, torch.device("cpu"))
        trainer.train()
        trainer.load_trained_agents()
        handler = OnlineDriftHandler(
            cfg, data, model, trainer.selector, trainer.collector_for,
            trainer.env, torch.device("cpu"), cfg.kpi(),
        )
        online = handler.run()
        assert np.isfinite(online.final_holdout["accuracy"])
        compute_baselines(
            cfg, trainer.env, trainer.selector, trainer.collector_for, cfg.kpi(),
        )
    finally:
        data.test_windows = original


def test_frozen_evaluation_does_not_update_model(data_bundle):
    cfg, data, _ = data_bundle
    model, _ = train_initial_predictor(data, cfg, torch.device("cpu"))
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    metrics = evaluate_frozen_model(cfg, data, model, torch.device("cpu"))
    after = model.state_dict()
    assert np.isfinite(metrics["accuracy"])
    assert model.training is False
    for key, value in before.items():
        assert torch.equal(value, after[key])


def test_end_to_end_smoke_outputs(data_bundle):
    cfg, data, _ = data_bundle
    model, _ = train_initial_predictor(data, cfg, torch.device("cpu"))
    trainer = RLTrainer(cfg, data, model, torch.device("cpu"))
    summary = trainer.train()
    assert summary["events"]
    assert (cfg.results_dir / "selector_q.json").exists()
    assert list((cfg.results_dir / "collectors").glob("*.json"))
    assert (cfg.results_dir / "drift_events.csv").exists()

    trainer.load_trained_agents()
    handler = OnlineDriftHandler(
        cfg, data, model, trainer.selector, trainer.collector_for,
        trainer.env, torch.device("cpu"), cfg.kpi(),
    )
    online = handler.run()
    assert online.alerts >= 1
    assert np.isfinite(online.final_holdout["accuracy"])
    test_metrics = evaluate_frozen_model(cfg, data, online.bank, torch.device("cpu"))
    assert np.isfinite(test_metrics["accuracy"])
    assert test_metrics["windows"] == len(data.test_windows)

    rows, aggregate = compute_baselines(
        cfg, trainer.env, trainer.selector, trainer.collector_for, cfg.kpi(),
    )
    assert rows
    assert aggregate
    assert any(row["method"] == "always_retrain" for row in rows)
