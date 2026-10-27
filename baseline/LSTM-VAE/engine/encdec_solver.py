"""Training and evaluation loop for the LSTM encoder-decoder (EncDec-AD)."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from Utils.io_utils import instantiate_from_config
from Utils.reconstruction_report import (
    fit_gaussian,
    mahalanobis_scores,
    select_threshold,
    write_reconstruction_report,
)


class EncDecTrainer:
    """Normal-only reconstruction training plus validation-selected scoring."""

    def __init__(self, config, args, model, dataloader=None, validation_info=None):
        self.config = config
        self.args = args
        self.model = model
        self.device = next(model.parameters()).device

        self.dataset = dataloader['dataset'] if dataloader is not None else None
        self.train_loader = dataloader['dataloader'] if dataloader is not None else None
        self.validation_info = None
        self.validation_dataset = None
        self.validation_loader = None
        self.fit_loader = None
        self.validation_fit_indices = np.asarray([], dtype=np.int64)
        self.validation_selection_indices = np.asarray([], dtype=np.int64)

        solver = config['solver']
        dataloader_config = config['dataloader']
        self.max_epochs = int(solver['max_epochs'])
        if self.max_epochs <= 0:
            raise ValueError('max_epochs must be positive.')
        self.gradient_accumulate_every = int(
            solver.get('gradient_accumulate_every', 1)
        )
        if self.gradient_accumulate_every <= 0:
            raise ValueError('gradient_accumulate_every must be positive.')
        self.early_stopping_patience = int(
            solver.get('early_stopping_patience', 30)
        )
        if self.early_stopping_patience < 0:
            raise ValueError('early_stopping_patience must be non-negative.')
        self.validation_fit_ratio = float(
            dataloader_config.get('validation_fit_ratio', 0.5)
        )
        if not 0.0 <= self.validation_fit_ratio <= 1.0:
            raise ValueError('validation_fit_ratio must be in [0, 1].')
        self.beta = float(dataloader_config.get('beta', 0.05))
        self.covariance_regularization = float(
            dataloader_config.get('covariance_regularization', 1.0e-3)
        )
        self.fallback_false_positive_rate = float(
            dataloader_config.get('fallback_false_positive_rate', 0.01)
        )
        configured = dataloader_config.get('decision_threshold')
        self.configured_threshold = (
            float(configured)
            if configured is not None and np.isfinite(float(configured))
            else None
        )

        self.results_folder = Path(solver['results_folder'])
        self.results_folder.mkdir(parents=True, exist_ok=True)

        base_lr = float(solver.get('base_lr', 1.0e-3))
        self.opt = Adam(
            filter(lambda parameter: parameter.requires_grad, self.model.parameters()),
            lr=base_lr,
            betas=(0.9, 0.999),
        )
        scheduler = solver.get('scheduler')
        if scheduler is not None:
            scheduler_config = deepcopy(scheduler)
            scheduler_config.setdefault('params', {})['optimizer'] = self.opt
            self.sch = instantiate_from_config(scheduler_config)
        else:
            self.sch = None

        self.epoch = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.milestone = 'best'
        self.decision_threshold = None
        self.last_detection_report = None
        self.detection_records = []

        if validation_info is not None:
            self.configure_validation(validation_info)

    # ------------------------------------------------------------------ data
    def configure_validation(self, validation_info):
        dataset = validation_info['dataset']
        if getattr(dataset, 'task', None) != 'reconstruction':
            raise ValueError('EncDecTrainer requires a reconstruction dataset.')
        if not hasattr(dataset, 'window_labels'):
            raise ValueError('The validation dataset must expose window_labels.')

        self.validation_info = validation_info
        self.validation_dataset = dataset
        self.validation_loader = validation_info['dataloader']

        labels = np.asarray(dataset.window_labels, dtype=np.int64)
        normal = np.flatnonzero(labels == 0)
        anomalous = np.flatnonzero(labels != 0)
        if normal.size == 0:
            raise ValueError(
                'At least one fault-free validation window is required to fit '
                'the reconstruction-error distribution.'
            )

        if normal.size > 1:
            n_fit = int(round(normal.size * self.validation_fit_ratio))
            n_fit = min(max(n_fit, 1), normal.size - 1)
        else:
            n_fit = 1
        self.validation_fit_indices = normal[:n_fit]
        selection = np.concatenate([normal[n_fit:], anomalous])
        if selection.size == 0:
            selection = np.arange(len(labels), dtype=np.int64)
        self.validation_selection_indices = np.sort(selection).astype(np.int64)

        batch_size = int(
            validation_info['dataloader'].batch_size
            if validation_info.get('dataloader') is not None
            else self.config['dataloader']['batch_size']
        )
        self.fit_loader = DataLoader(
            Subset(dataset, self.validation_fit_indices.tolist()),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.device.type == 'cuda',
            drop_last=False,
        )

    # -------------------------------------------------------------- protocol
    def _reference_dataset(self):
        if self.validation_dataset is not None:
            return self.validation_dataset
        return self.dataset

    def _protocol(self, dataset=None):
        dataset = dataset if dataset is not None else self._reference_dataset()
        if dataset is None:
            return None
        if getattr(dataset, 'task', None) != 'reconstruction':
            raise ValueError('Checkpoints are only written for reconstruction datasets.')
        return {
            'task': 'reconstruction',
            'window': int(dataset.window),
            'DeltaT': int(dataset.DeltaT),
            'feature_size': int(self.model.feature_size),
            'model': {
                'hidden_size': int(self.model.hidden_size),
                'num_layers': int(self.model.num_layers),
                'dropout': float(self.model.dropout),
            },
            'split_timestamp': int(dataset.split_timestamp),
            'validation_timestamp': int(dataset.validation_timestamp),
            'normalization': dataset.normalization_state(),
        }

    def _validate_protocol(self, saved):
        current = self._protocol()
        if current is None or saved is None:
            return
        for key in (
            'task',
            'window',
            'DeltaT',
            'feature_size',
            'model',
            'split_timestamp',
            'validation_timestamp',
            'normalization',
        ):
            if saved.get(key) != current.get(key):
                raise ValueError(
                    f'Checkpoint protocol mismatch for {key}: '
                    f'saved {saved.get(key)!r}, current {current.get(key)!r}.'
                )

    # ------------------------------------------------------------- checkpoints
    def save(self, milestone, verbose=False):
        payload = {
            'model': self.model.state_dict(),
            'optimizer': self.opt.state_dict(),
            'scheduler': self.sch.state_dict() if self.sch is not None else None,
            'epoch': int(self.epoch),
            'best_loss': float(self.best_loss),
            'protocol': self._protocol(),
        }
        path = self.results_folder / f'checkpoint-{milestone}.pt'
        torch.save(payload, str(path))
        if verbose:
            print(f'Saved checkpoint to {path}')

    def load(self, milestone='best', verbose=False):
        path = self.results_folder / f'checkpoint-{milestone}.pt'
        if not path.is_file():
            raise FileNotFoundError(f'No checkpoint found at {path}.')
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)

        self._validate_protocol(payload.get('protocol'))
        self.model.load_state_dict(payload['model'])
        if payload.get('optimizer') is not None:
            self.opt.load_state_dict(payload['optimizer'])
        self.epoch = int(payload.get('epoch', 0))
        self.best_loss = float(payload.get('best_loss', float('inf')))
        if self.sch is not None and payload.get('scheduler') is not None:
            try:
                self.sch.load_state_dict(payload['scheduler'])
            except Exception as error:  # scheduler versions differ across runs
                print(f'Warning: could not restore scheduler state ({error}).')
        self.milestone = milestone
        if verbose:
            print(f'Loaded checkpoint from {path}')

    # ---------------------------------------------------------------- training
    def _validation_loss(self):
        if self.fit_loader is None:
            raise ValueError('Training requires a validation split.')
        self.model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for x, _ in self.fit_loader:
                x = x.to(self.device)
                reconstruction = self.model(x, teacher_forcing=False)
                total += F.mse_loss(reconstruction, x, reduction='sum').item()
                count += x.numel()
        self.model.train()
        if count == 0:
            raise ValueError('The validation split is empty.')
        return total / count

    def train(self):
        if self.train_loader is None:
            raise ValueError('No training dataloader was provided.')
        if self.fit_loader is None:
            raise ValueError(
                'Training requires a validation split for early stopping and '
                'reconstruction-error distribution fitting.'
            )
        if len(self.train_loader) == 0:
            raise ValueError('The training split is empty.')

        for epoch in range(self.epoch, self.max_epochs):
            self.model.train()
            total, count = 0.0, 0
            for step, (x, _) in enumerate(self.train_loader):
                x = x.to(self.device)
                raw_loss = F.mse_loss(
                    self.model(x, teacher_forcing=True), x
                )
                (raw_loss / self.gradient_accumulate_every).backward()
                total += raw_loss.item() * x.shape[0]
                count += x.shape[0]
                if (step + 1) % self.gradient_accumulate_every == 0:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.opt.step()
                    self.opt.zero_grad()

            if len(self.train_loader) % self.gradient_accumulate_every != 0:
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.opt.zero_grad()

            self.epoch = epoch + 1
            train_loss = total / max(count, 1)
            validation_loss = self._validation_loss()
            if not math.isfinite(validation_loss):
                raise ValueError('Validation loss became non-finite.')
            if self.sch is not None:
                self.sch.step(validation_loss)

            improved = validation_loss < self.best_loss
            if improved:
                self.best_loss = validation_loss
                self.patience_counter = 0
                self.save('best')
            else:
                self.patience_counter += 1

            learning_rate = self.opt.param_groups[0]['lr']
            print(
                f'epoch {self.epoch}/{self.max_epochs} '
                f'train_mse {train_loss:.6f} '
                f'validation_mse {validation_loss:.6f} '
                f'lr {learning_rate:.3e}'
            )

            if (
                self.patience_counter > self.early_stopping_patience
                and self.early_stopping_patience >= 0
            ):
                print(
                    f'Early stopping after {self.patience_counter} epochs without '
                    'validation improvement.'
                )
                break

        print('training complete')

    # ---------------------------------------------------------------- scoring
    def fit_error_distribution(self):
        if self.fit_loader is None:
            raise ValueError('No validation fit loader is available.')
        self.model.eval()
        collected = []
        with torch.no_grad():
            for x, _ in self.fit_loader:
                x = x.to(self.device)
                reconstruction = self.model(x, teacher_forcing=False)
                errors = (x - reconstruction).abs().cpu().numpy()
                collected.append(errors.reshape(-1, errors.shape[-1]))
        if not collected:
            raise ValueError('The validation fit split is empty.')
        errors = np.concatenate(collected, axis=0)
        return fit_gaussian(errors, regularization=self.covariance_regularization)

    def score_dataset(self, dataset, dataloader, stats):
        if getattr(dataset, 'task', None) != 'reconstruction':
            raise ValueError('Scoring requires a reconstruction dataset.')
        self.model.eval()
        scores = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.to(self.device)
                reconstruction = self.model(x, teacher_forcing=False)
                errors = (x - reconstruction).abs().cpu().numpy()
                scores.append(mahalanobis_scores(errors, stats))
        if not scores:
            raise ValueError('Cannot score an empty split.')
        point_scores = np.concatenate(scores, axis=0)
        if len(point_scores) != len(dataset.sample_starts):
            raise ValueError(
                'The dataloader did not return exactly one reconstruction per window.'
            )
        return point_scores

    def evaluate(self, validation_info=None, test_info=None):
        if validation_info is not None:
            self.configure_validation(validation_info)
        if self.validation_dataset is None:
            raise ValueError('Evaluation requires a validation dataset.')

        checkpoint = getattr(self.args, 'checkpoint', 'best')
        self.load(checkpoint)
        stats = self.fit_error_distribution()

        validation_scores = self.score_dataset(
            self.validation_dataset, self.validation_loader, stats
        )
        selection_scores = validation_scores[self.validation_selection_indices]
        selection_labels = np.asarray(self.validation_dataset.labels)[
            self.validation_selection_indices
        ]
        selection = select_threshold(
            selection_scores,
            selection_labels,
            beta=self.beta,
            configured_threshold=self.configured_threshold,
            fallback_false_positive_rate=self.fallback_false_positive_rate,
        )
        threshold = float(selection['selected_threshold'])
        self.decision_threshold = threshold

        checkpoint_path = str(self.results_folder / f'checkpoint-{self.milestone}.pt')
        validation_summary = write_reconstruction_report(
            dataset=self.validation_dataset,
            point_scores=validation_scores,
            threshold=threshold,
            beta=self.beta,
            output_dir=self.args.save_dir,
            split='validation',
            config=self.config['dataloader'],
            stats=stats,
            checkpoint=checkpoint_path,
            checkpoint_step=self.epoch,
            threshold_selection=selection,
        )
        self.last_detection_report = validation_summary
        self._print_summary('validation', validation_summary)

        if test_info is not None:
            test_dataset = test_info['dataset']
            test_scores = self.score_dataset(
                test_dataset, test_info['dataloader'], stats
            )
            test_summary = write_reconstruction_report(
                dataset=test_dataset,
                point_scores=test_scores,
                threshold=threshold,
                beta=self.beta,
                output_dir=self.args.save_dir,
                split='test',
                config=self.config['dataloader'],
                stats=stats,
                checkpoint=checkpoint_path,
                checkpoint_step=self.epoch,
                threshold_selection=None,
            )
            self.last_detection_report = test_summary
            self._print_summary('test', test_summary)

        return self.last_detection_report

    @staticmethod
    def _print_summary(split, summary):
        metrics = summary['metrics']
        point = metrics['point']
        window = metrics['window']
        print(
            f'{split} point-level pre:{point["precision"]:.6f},'
            f'rec:{point["recall"]:.6f},fbeta:{point["fbeta"]:.6f}; '
            f'window-level pre:{window["precision"]:.6f},'
            f'rec:{window["recall"]:.6f},fbeta:{window["fbeta"]:.6f}; '
            f'threshold:{summary["threshold"]:.17g}'
        )
