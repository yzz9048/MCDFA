"""RL-based automated concept drift handling (paper reproduction).

Implements the framework of Shayesteh et al., "Automated Concept Drift
Handling for Fault Prediction in Edge Clouds Using Reinforcement Learning"
(IEEE TNSM 2022), adapted to this repository's datasets:

* supervised CNN fault predictor (paper's chosen model, 12 -> 6 horizon),
* fixed-threshold drift detector on natural chronological data,
* Q-learning Drift Adaptation Method Selector (layer-wise R/F/Z/P/D),
* Q-learning Data Collector (increase/decrease data size).

Test data is never used for drift detection, adaptation, rewards, or Q-table
updates (strict no-leakage mode).
"""

from .config import Config, load_config
from .data import PaperDataModule
from .model import FaultCNN, ModelBank
from .drift_detector import DriftDetector, DriftEvent, find_natural_drift_events
from .adaptation import AdaptationEnvironment, AdaptationPlan
from .q_learning import MethodSelectorAgent, DataCollectorAgent, EpsilonSchedule
from .trainer import RLTrainer
from .protocol import OnlineDriftHandler, evaluate_frozen_model

__all__ = [
    'Config', 'load_config', 'PaperDataModule', 'FaultCNN', 'ModelBank',
    'DriftDetector', 'DriftEvent', 'find_natural_drift_events',
    'AdaptationEnvironment', 'AdaptationPlan',
    'MethodSelectorAgent', 'DataCollectorAgent', 'EpsilonSchedule',
    'RLTrainer', 'OnlineDriftHandler', 'evaluate_frozen_model',
]
