"""Configuration for the RL-based concept-drift framework.

The defaults encode the paper's public settings:

* predictor: CNN, input window 12, output horizon 6, 10 RFE features;
* method selector: 12,000 episodes, epsilon 1 -> 0.1 after 2,000 episodes,
  decaying by 0.1 every 200 episodes;
* data collector: 20,000 episodes, data sizes 1,000..7,000 in steps of 100;
* rewards: k1, k2, k3 = 0.8, 0.1, 0.1;
* KPI requirement sets from Table II.

Parameters the paper does not specify (alpha, gamma, replay size, drift
detector threshold, episode length) have explicit, documented defaults and
can be overridden in YAML.  ``smoke_test`` shrinks every expensive setting.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class PredictorConfig:
    input_window: int = 12
    forecast_horizon: int = 6
    rfe_features: int = 10
    conv1_out: int = 32
    conv2_out: int = 64
    kernel_size: int = 3
    pool_size: int = 2
    dense1: int = 128
    dropout: float = 0.1
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    # Number of tail timestamps inside the 60% train segment used for early
    # stopping.  Never taken from validation/test.
    val_fraction: float = 0.2
    patience: int = 8


@dataclass
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    # The validation segment is split chronologically: the first part is the
    # natural drift stream used by the detector / adaptation; the tail is a
    # fixed accuracy holdout used only for rewards (never for test).
    rl_stream_fraction: float = 0.8


@dataclass
class DetectorConfig:
    rolling_window: int = 100
    accuracy_threshold: float = 0.7
    cooldown: int = 100
    min_events: int = 5
    max_events: int = 12
    min_post_windows: int = 1000
    # If the fixed threshold yields fewer than min_events, use the lowest
    # natural rolling-accuracy minima as drift candidates.  This is still a
    # natural-drift protocol: no synthetic drift is injected.
    fallback_to_accuracy_minima: bool = True


@dataclass
class KPISet:
    name: str
    time: float
    resource: float
    accuracy: float


@dataclass
class RewardConfig:
    k1: float = 0.8
    k2: float = 0.1
    k3: float = 0.1


@dataclass
class QLearningConfig:
    # The paper does not publish alpha/gamma/replay size; these are explicit
    # engineering defaults.
    alpha: float = 0.1
    gamma: float = 0.9
    replay_size: int = 64
    replay_capacity: int = 2_000
    selector_episodes: int = 12_000
    collector_episodes: int = 20_000
    explore_episodes: int = 2_000
    decay_every: int = 200
    decay_amount: float = 0.1
    min_epsilon: float = 0.1
    # Paper does not specify episode length; interpreted as the number of
    # increase/decrease decisions the data collector may make.
    collector_episode_length: int = 10
    # "max" uses all post-drift data available for each event, capped by Dr.
    selector_data_size: Any = "max"


@dataclass
class DataCollectorConfig:
    D0: int = 1_000
    Dr: int = 7_000
    interval: int = 100
    deployment_initial_size: int = 4_000


@dataclass
class AdaptationConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    cache: bool = True
    # Cap adaptation data per event by the data collector's Dr.
    max_data_size: int = 7_000
    # The paper does not define how C is measured.  "footprint" uses the
    # deterministic bytes of adaptation data + model + one batch of
    # activations; "measured" uses peak RSS/CUDA allocation.
    resource_metric: str = "footprint"


@dataclass
class Config:
    seed: int = 12345
    device: str = "auto"
    output_root: str = "RESULTS"
    dataset: str = "sn"
    data_root: str = "Data/sn.csv"
    active_kpi: str = "cpu"
    smoke_test: bool = False
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    kpi_sets: List[KPISet] = field(default_factory=lambda: [
        KPISet("congestion", 50.0, 10.0, 0.86),
        KPISet("hdd", 25.0, 5.0, 0.83),
        KPISet("packet_loss", 15.0, 9.0, 0.89),
        KPISet("cpu", 10.0, 7.0, 0.90),
    ])
    reward: RewardConfig = field(default_factory=RewardConfig)
    qlearning: QLearningConfig = field(default_factory=QLearningConfig)
    data_collector: DataCollectorConfig = field(default_factory=DataCollectorConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

    @property
    def results_dir(self) -> Path:
        suffix = "_smoke" if self.smoke_test else ""
        return Path(self.output_root) / f"rl_paper_{self.dataset}{suffix}"

    def kpi(self, name: Optional[str] = None) -> KPISet:
        wanted = (name or self.active_kpi).casefold()
        for item in self.kpi_sets:
            if item.name.casefold() == wanted:
                return item
        names = ", ".join(item.name for item in self.kpi_sets)
        raise ValueError(f"Unknown KPI set {wanted!r}; choose one of: {names}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_DATA_ROOTS = {
    "sn": "Data/sn.csv",
    "media": "Data/media.csv",
    # The repository's MAAT/AIOps dataset is stored as AIOps-R.csv.
    "maat": "Data/AIOps-R.csv",
    "aiops": "Data/AIOps-R.csv",
    "gaia": "Data/GAIA-R.csv",
}

_SMOKE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "predictor": {
        "epochs": 2,
        "batch_size": 32,
        "patience": 2,
        "val_fraction": 0.25,
    },
    "split": {"rl_stream_fraction": 0.7},
    "detector": {
        "rolling_window": 10,
        "accuracy_threshold": 0.9,
        "cooldown": 10,
        "min_events": 2,
        "max_events": 3,
        "min_post_windows": 50,
    },
    "qlearning": {
        "selector_episodes": 40,
        "collector_episodes": 40,
        "explore_episodes": 10,
        "decay_every": 5,
        "decay_amount": 0.2,
        "replay_size": 8,
        "collector_episode_length": 3,
    },
    "adaptation": {"epochs": 2, "batch_size": 32},
    "data_collector": {
        "D0": 100,
        "Dr": 500,
        "interval": 100,
        "deployment_initial_size": 300,
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _construct(section_type, values: Dict[str, Any]):
    allowed = {item.name for item in fields(section_type)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {section_type.__name__} keys: {sorted(unknown)}."
        )
    return section_type(**values)


def _kpi_from_dict(values: Dict[str, Any]) -> KPISet:
    allowed = {"name", "time", "resource", "accuracy"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown KPI keys: {sorted(unknown)}.")
    return KPISet(**values)


def load_config(
    path: Optional[str] = None,
    dataset: Optional[str] = None,
    smoke_test: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load YAML with defaults and apply smoke-test/CLI overrides."""
    raw: Dict[str, Any] = {}
    if path is not None:
        with open(path, encoding="utf-8-sig") as handle:
            raw = yaml.safe_load(handle) or {}
    raw = copy.deepcopy(raw)

    if dataset is not None:
        raw["dataset"] = dataset
    effective_smoke = bool(smoke_test or raw.get("smoke_test", False))
    if effective_smoke:
        _deep_update(raw, copy.deepcopy(_SMOKE_OVERRIDES))
    if overrides:
        _deep_update(raw, overrides)
    dataset_name = str(raw.get("dataset", "sn")).casefold()

    datasets = raw.pop("datasets", {}) or {}
    data_root = raw.pop("data_root", None)
    if data_root is None:
        data_root = (datasets.get(dataset_name) or {}).get("data_root")
    if data_root is None:
        data_root = (datasets.get(str(raw.get("dataset", "sn"))) or {}).get("data_root")
    if data_root is None:
        data_root = DEFAULT_DATA_ROOTS.get(dataset_name)
    if data_root is None:
        raise ValueError(
            f"No data root configured for dataset {dataset_name!r}. "
            f"Known defaults: {sorted(DEFAULT_DATA_ROOTS)}."
        )

    section_defaults = {
        "predictor": PredictorConfig,
        "split": SplitConfig,
        "detector": DetectorConfig,
        "reward": RewardConfig,
        "qlearning": QLearningConfig,
        "data_collector": DataCollectorConfig,
        "adaptation": AdaptationConfig,
    }
    sections = {}
    for name, section_type in section_defaults.items():
        sections[name] = _construct(section_type, raw.pop(name, {}) or {})

    kpi_raw = raw.pop("kpi_sets", None)
    kpi_sets = (
        [_kpi_from_dict(item) for item in kpi_raw]
        if kpi_raw is not None else Config.__dataclass_fields__["kpi_sets"].default_factory()
    )
    known = {
        "seed", "device", "output_root", "dataset", "active_kpi",
        "smoke_test", "data_root",
    }
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown top-level config keys: {sorted(unknown)}.")

    cfg = Config(
        seed=int(raw.get("seed", 12345)),
        device=str(raw.get("device", "auto")),
        output_root=str(raw.get("output_root", "RESULTS")),
        dataset=str(raw.get("dataset", dataset_name)),
        data_root=str(data_root),
        active_kpi=str(raw.get("active_kpi", "cpu")),
        smoke_test=effective_smoke,
        kpi_sets=kpi_sets,
        **sections,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: Config) -> None:
    p = cfg.predictor
    if p.input_window <= 0 or p.forecast_horizon <= 0:
        raise ValueError("Predictor windows must be positive.")
    if p.rfe_features <= 0:
        raise ValueError("rfe_features must be positive.")
    if p.conv1_out <= 0 or p.conv2_out <= 0 or p.dense1 <= 0:
        raise ValueError("CNN layer widths must be positive.")
    if p.kernel_size <= 0 or p.pool_size <= 0:
        raise ValueError("CNN kernel_size and pool_size must be positive.")
    conv_len = p.input_window - 2 * (p.kernel_size - 1)
    if conv_len < p.pool_size:
        raise ValueError(
            "input_window is too short for the CNN kernel and pooling sizes."
        )
    if not 0.0 <= p.dropout < 1.0:
        raise ValueError("predictor.dropout must be in [0, 1).")
    if p.epochs <= 0 or p.batch_size <= 0 or p.lr <= 0:
        raise ValueError("Predictor epochs, batch_size, and lr must be positive.")
    if p.patience <= 0:
        raise ValueError("predictor.patience must be positive.")
    if not 0 < p.val_fraction < 1:
        raise ValueError("predictor.val_fraction must be in (0, 1).")

    total = cfg.split.train + cfg.split.validation + cfg.split.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split proportions must sum to 1, got {total}.")
    if min(cfg.split.train, cfg.split.validation, cfg.split.test) <= 0:
        raise ValueError("Each split proportion must be positive.")
    if not 0 < cfg.split.rl_stream_fraction < 1:
        raise ValueError("rl_stream_fraction must be in (0, 1).")

    detector = cfg.detector
    if detector.rolling_window < 2:
        raise ValueError("detector.rolling_window must be >= 2.")
    if not 0.0 <= detector.accuracy_threshold <= 1.0:
        raise ValueError("detector.accuracy_threshold must be in [0, 1].")
    if detector.cooldown < 1:
        raise ValueError("detector.cooldown must be positive.")
    if detector.min_events < 1 or detector.max_events < detector.min_events:
        raise ValueError("Require 1 <= detector.min_events <= max_events.")
    if detector.min_post_windows < 1:
        raise ValueError("detector.min_post_windows must be positive.")

    collector = cfg.data_collector
    if collector.D0 <= 0 or collector.Dr < collector.D0:
        raise ValueError("Require 0 < D0 <= Dr.")
    if collector.interval <= 0:
        raise ValueError("data_collector.interval must be positive.")
    if collector.deployment_initial_size <= 0:
        raise ValueError("data_collector.deployment_initial_size must be positive.")

    q = cfg.qlearning
    if q.selector_episodes <= 0 or q.collector_episodes <= 0:
        raise ValueError("Q-learning episode counts must be positive.")
    if q.replay_size <= 0 or q.replay_capacity <= 0:
        raise ValueError("replay_size and replay_capacity must be positive.")
    if q.collector_episode_length <= 0:
        raise ValueError("collector_episode_length must be positive.")
    if q.explore_episodes < 0 or q.decay_every <= 0 or q.decay_amount < 0:
        raise ValueError("Invalid epsilon schedule (explore/decay).")
    if not 0 <= q.min_epsilon <= 1:
        raise ValueError("min_epsilon must be in [0, 1].")
    if not 0 < q.gamma <= 1:
        raise ValueError("gamma must be in (0, 1].")
    if not 0 < q.alpha <= 1:
        raise ValueError("alpha must be in (0, 1].")
    if q.selector_data_size != "max":
        if int(q.selector_data_size) <= 0:
            raise ValueError("selector_data_size must be 'max' or positive.")

    if cfg.reward.k1 < 0 or cfg.reward.k2 < 0 or cfg.reward.k3 < 0:
        raise ValueError("Reward coefficients must be non-negative.")

    a = cfg.adaptation
    if a.epochs <= 0 or a.batch_size <= 0 or a.lr <= 0:
        raise ValueError("Adaptation epochs, batch_size, and lr must be positive.")
    if a.max_data_size <= 0:
        raise ValueError("adaptation.max_data_size must be positive.")
    if a.resource_metric not in {"footprint", "measured"}:
        raise ValueError("adaptation.resource_metric must be footprint or measured.")

    if not cfg.kpi_sets:
        raise ValueError("At least one KPI set is required.")
    kpi_names = [kpi.name.casefold() for kpi in cfg.kpi_sets]
    if len(set(kpi_names)) != len(kpi_names):
        raise ValueError("KPI set names must be unique.")
    for kpi in cfg.kpi_sets:
        if kpi.time <= 0 or kpi.resource <= 0:
            raise ValueError(f"KPI {kpi.name} requires positive time/resource.")
        if not 0 <= kpi.accuracy <= 1:
            raise ValueError(f"KPI {kpi.name} accuracy must be in [0, 1].")
    cfg.kpi()  # validate active_kpi
    if not Path(cfg.data_root).exists():
        raise FileNotFoundError(
            f"Dataset not found: {cfg.data_root}. "
            "Set data_root in config/rl_paper.yaml or pass --data-root."
        )
