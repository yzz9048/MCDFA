"""Entry point for the RL-based concept-drift framework.

Stages
------
train-predictor : paper CNN on the 60% train segment, early-stopped on its
                  internal tail.
train-rl        : natural drift detection on the validation stream, then the
                  Q-learning Method Selector and Data Collector.
evaluate        : Algorithm 1 on the validation stream, frozen test evaluation,
                  and validation-only baseline comparisons.
all             : all three stages in order (default).

Examples
--------
    python main.py --dataset sn --smoke-test
    python main.py --dataset sn --stage train-predictor
    python main.py --dataset sn --stage train-rl
    python main.py --dataset sn --stage evaluate
    python main.py --dataset media --active-kpi hdd
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from rl_drift.baselines import compute_baselines, save_baselines
from rl_drift.config import load_config
from rl_drift.data import PaperDataModule
from rl_drift.protocol import (
    OnlineDriftHandler, evaluate_frozen_model, save_online_result,
)
from rl_drift.trainer import (
    RLTrainer, load_predictor, save_predictor, train_initial_predictor,
)
from rl_drift.utils import resolve_device, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "rl_paper.yaml"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="RL-based automated concept drift handling (paper reproduction)"
    )
    parser.add_argument("--dataset", type=str, default="sn",
                        help="Dataset key: sn, media, maat/aiops, gaia.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="YAML config path.")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Override the dataset CSV path.")
    parser.add_argument(
        "--stage",
        choices=["all", "train-predictor", "train-rl", "evaluate"],
        default="all",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Shrink all expensive settings for a fast run.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None,
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default=None,
                        help="Output root (default RESULTS).")
    parser.add_argument("--active-kpi", type=str, default=None,
                        help="KPI set used for online adaptation (default cpu).")
    parser.add_argument("--predictor", type=str, default=None,
                        help="Predictor checkpoint path.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable adaptation-result memoisation.")
    parser.add_argument("--skip-baselines", action="store_true",
                        help="Skip validation baseline comparisons.")
    parser.add_argument("--reuse-predictor", action="store_true",
                        help="With --stage all, reuse an existing predictor.pt.")
    return parser.parse_args(argv)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _config_from_args(args):
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.device is not None:
        overrides["device"] = args.device
    if args.output is not None:
        overrides["output_root"] = args.output
    if args.active_kpi is not None:
        overrides["active_kpi"] = args.active_kpi
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.no_cache:
        overrides.setdefault("adaptation", {})["cache"] = False
    return load_config(
        args.config,
        dataset=args.dataset,
        smoke_test=args.smoke_test,
        overrides=overrides,
    )


def _prepare(config, device):
    _seed_all(config.seed)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    write_json(config.results_dir / "config.json", config.to_dict())
    data = PaperDataModule(config)
    write_json(config.results_dir / "data_summary.json", data.summary())
    data.save_preprocessing(config.results_dir / "preprocessing.npz")
    print(
        f"[data] {config.data_root}: timestamps={data.n_timestamps} "
        f"features={len(data.feature_names)}->{len(data.selected_indices)} "
        f"fit={len(data.fit_windows)} stream={len(data.stream_windows)} "
        f"holdout={len(data.accuracy_windows)} test={len(data.test_windows)}"
    )
    return data


def _predictor_path(args, config) -> Path:
    return Path(args.predictor) if args.predictor else config.results_dir / "predictor.pt"


def _train_predictor(args, config, data, device):
    path = _predictor_path(args, config)
    model, rows = train_initial_predictor(
        data, config, device,
        log_path=config.results_dir / "predictor_training.csv",
    )
    save_predictor(path, model)
    last = rows[-1] if rows else {}
    print(
        f"[predictor] saved {path} | epochs={len(rows)} "
        f"val_acc={last.get('predictor_val_accuracy')}"
    )
    return model, path


def _load_or_train_predictor(args, config, data, device):
    path = _predictor_path(args, config)
    if path.exists():
        model = load_predictor(path, device)
        print(f"[predictor] loaded {path}")
        return model
    return _train_predictor(args, config, data, device)[0]


def _run_online_and_test(config, data, predictor, trainer, device):
    trainer.load_trained_agents()
    handler = OnlineDriftHandler(
        config=config,
        data=data,
        initial_model=predictor,
        selector=trainer.selector,
        collector_lookup=trainer.collector_for,
        environment=trainer.env,
        device=device,
        kpi=config.kpi(),
    )
    online = handler.run()
    save_online_result(config.results_dir, online)
    print(
        f"[online] alerts={online.alerts} consumed={online.consumed_windows} "
        f"holdout {online.initial_holdout['accuracy']:.4f} -> "
        f"{online.final_holdout['accuracy']:.4f}"
    )

    test_initial = evaluate_frozen_model(config, data, predictor, device)
    test_final = evaluate_frozen_model(config, data, online.bank, device)
    test_payload = {
        "protocol": "strict_no_test_leakage",
        "initial_predictor": test_initial,
        "after_validation_only_adaptation": test_final,
    }
    write_json(config.results_dir / "test_metrics.json", test_payload)
    print(
        f"[test] initial acc={test_initial['accuracy']:.4f} "
        f"balanced={test_initial['balanced_accuracy']:.4f} | "
        f"adapted acc={test_final['accuracy']:.4f} "
        f"balanced={test_final['balanced_accuracy']:.4f}"
    )

    baseline_aggregate = []
    if not config.skip_baselines:
        rows, baseline_aggregate = compute_baselines(
            config, trainer.env, trainer.selector, trainer.collector_for,
            config.kpi(),
        )
        save_baselines(config.results_dir, rows, baseline_aggregate)
        print("[baselines]")
        for row in baseline_aggregate:
            print(
                f"  {row['method']:<26} reward={row['mean_reward']:+.4f} "
                f"acc={row['mean_accuracy']:.4f}"
            )

    summary = {
        "config": config.to_dict(),
        "data": data.summary(),
        "drift_events": trainer.event_summary(),
        "online": online.summary(),
        "test": test_payload,
        "baselines": baseline_aggregate,
    }
    write_json(config.results_dir / "evaluation_summary.json", summary)
    return summary


def main(argv=None):
    args = parse_args(argv)
    config = _config_from_args(args)
    config.skip_baselines = bool(args.skip_baselines)
    device = resolve_device(config.device)
    print(
        f"[config] dataset={config.dataset} stage={args.stage} "
        f"smoke={config.smoke_test} device={device} "
        f"results={config.results_dir}"
    )
    data = _prepare(config, device)

    if args.stage == "train-predictor":
        _train_predictor(args, config, data, device)
        return

    predictor_path = _predictor_path(args, config)
    if args.stage == "all" and args.reuse_predictor and predictor_path.exists():
        predictor = load_predictor(predictor_path, device)
        print(f"[predictor] reused {predictor_path}")
    elif args.stage in {"all", "train-predictor"}:
        predictor = _train_predictor(args, config, data, device)[0]
    else:
        predictor = _load_or_train_predictor(args, config, data, device)

    trainer = RLTrainer(config, data, predictor, device=device)
    if args.stage in {"all", "train-rl"}:
        trainer.train()
    if args.stage in {"all", "evaluate"}:
        _run_online_and_test(config, data, predictor, trainer, device)


if __name__ == "__main__":
    main()
