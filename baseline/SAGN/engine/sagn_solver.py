"""Offline training and online testing for SAGN.

The trainer implements the two stages of SAGN:

Offline (Section 4.1)
    * sampled correlation graph ``G0`` via Gumbel-Softmax,
    * seq2seq DCGRU training with MAE,
    * a pool of ``C`` well-performing snapshots selected on validation data.

Online (Section 4.2)
    * every newly observed sample updates the current graph/network,
    * one-step residuals of already observed timestamps produce a causal,
      label-free alarm score for the detection protocol,
    * a dynamic validation set ``V*`` of size ``Bo`` periodically revalidates
      the current network and the pool, rolling the best one in.

The repository's 6/2/2 chronological split, 10-step horizon and detection
reporting contract are preserved: ``restore`` writes the same windows scores,
threshold selection and event reports as the previous predictor.
"""

import copy
import json
import random
from collections import deque
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from Utils.io_utils import instantiate_from_config
from Utils.detection_report import write_detection_report


def cycle(dataloader):
    while True:
        for batch in dataloader:
            yield batch


def _clone_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


class SAGNTrainer(object):
    def __init__(self, config, args, model, dataloader, validation=None, logger=None):
        self.config = config
        self.args = args
        self.model = model
        self.logger = logger
        self.device = next(model.parameters()).device
        self.train_info = dataloader
        self.dataset = dataloader['dataset'] if dataloader is not None else None
        self.train_loader = dataloader['dataloader'] if dataloader is not None else None
        self.validation_info = validation
        self.validation_dataset = (
            validation['dataset'] if validation is not None else None
        )
        self.dl = cycle(self.train_loader) if self.train_loader is not None else None
        self.step = 0
        self.milestone = 0
        self.train_num_steps = int(config['solver']['max_epochs'])
        self.gradient_accumulate_every = max(
            1, int(config['solver'].get('gradient_accumulate_every', 1))
        )
        self.results_folder = Path(
            str(config['solver']['results_folder']) + f'_{model.seq_length}'
        )
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.pool_size = max(1, int(model.pool_config.get('size', 20)))
        self.snapshot_interval = max(
            1, int(model.pool_config.get('snapshot_interval', 250))
        )
        self.validation_windows = max(
            1, int(model.pool_config.get('validation_windows', 256))
        )
        self.calibration_windows = max(
            1, int(model.residual_config.get('calibration_windows', 2048))
        )
        self.residual_steps = max(1, int(model.residual_config.get('steps', 10)))

        self.pool = []
        self.best_validation = float('inf')
        self.best_state = None
        self.best_step = None
        self.calibration = model.get_calibration()
        self.decision_threshold = float(
            config['dataloader'].get('decision_threshold', 0.5)
        )
        self.last_detection_report = None

        base_lr = float(config['solver'].get('base_lr', 1.0e-3))
        self.opt = Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.96))
        scheduler_config = copy.deepcopy(config['solver']['scheduler'])
        scheduler_config['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(scheduler_config)

        self._online_optimizer = None
        self._eval_model = None
        self._online_log = []

    # ------------------------------------------------------------------
    # protocol / persistence
    # ------------------------------------------------------------------
    def protocol(self):
        protocol = self.model.method_protocol()
        protocol['data'] = {
            'proportion': self.config['dataloader']['params'].get('proportion'),
            'validation_proportion': self.config['dataloader']['params'].get(
                'validation_proportion'
            ),
            'stride': self.config['dataloader']['params'].get('DeltaT'),
            'pred_len': self.args.pred_len,
        }
        return protocol

    def save(self, milestone, verbose=False):
        path = self.results_folder / f'checkpoint-{milestone}.pt'
        if verbose:
            print(f'Save current model to {path}')
        saved_step = (
            self.best_step
            if milestone == 'best' and self.best_step is not None
            else self.step
        )
        payload = {
            'step': saved_step,
            'model': self.model.state_dict(),
            'pool': [
                {
                    'state': entry['state'],
                    'loss': entry['loss'],
                    'step': entry['step'],
                }
                for entry in self.pool
            ],
            'calibration': self.calibration,
            'protocol': self.protocol(),
        }
        torch.save(payload, str(path))

    def load(self, milestone, verbose=False):
        path = self.results_folder / f'checkpoint-{milestone}.pt'
        if verbose:
            print(f'Resume from {path}')
        payload = torch.load(str(path), map_location=self.device)
        if payload.get('protocol') != self.protocol():
            raise ValueError(
                'The checkpoint does not match this SAGN configuration. '
                'Retrain the model with the current config.'
            )
        self.model.load_state_dict(payload['model'])
        self.calibration = payload.get('calibration') or self.model.get_calibration()
        self.model.set_calibration(self.calibration)
        self.pool = payload.get('pool', [])
        self.step = int(payload.get('step', 0))
        self.milestone = milestone
        self.model.eval()

    # ------------------------------------------------------------------
    # offline stage
    # ------------------------------------------------------------------
    def _batch_windows(self, dataset, count, samples=None):
        """Evenly spaced windows from a dataset, in chronological sample order."""
        total = len(dataset)
        if total == 0:
            raise ValueError('Cannot sample an empty dataset.')
        count = min(max(1, int(count)), total)
        indices = np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))
        source = dataset.samples if samples is None else samples
        windows = torch.as_tensor(
            source[indices], dtype=torch.float32, device=self.device
        )
        return windows, indices

    def _split_windows(self, windows):
        history = windows[:, : self.model.history_length, :]
        target = windows[:, self.model.history_length :, 1:]
        return history, target

    def _validation_windows(self):
        dataset = self.validation_dataset or self.dataset
        return self._batch_windows(dataset, self.validation_windows)

    def _validation_loss(self):
        """Autoregressive multi-step validation loss, as used at inference."""
        windows, _ = self._validation_windows()
        history, target = self._split_windows(windows)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            predictions = self.model.predict(history)
            loss = F.l1_loss(predictions, target).item()
        self.model.train(was_training)
        return loss

    def _consider_pool(self, validation_loss):
        entry = {
            'state': _clone_state(self.model.state_dict()),
            'loss': float(validation_loss),
            'step': self.step,
        }
        self.pool.append(entry)
        self.pool.sort(key=lambda item: item['loss'])
        if len(self.pool) > self.pool_size:
            self.pool = self.pool[: self.pool_size]

    def _fit_calibration(self):
        """Robust per-node residual calibration from label-free windows.

        The validation split is used when available (as agreed), because it is
        closer to the online regime; no fault label enters this step.
        """
        dataset = self.validation_dataset or self.dataset
        if dataset is None:
            return
        _, indices = self._batch_windows(dataset, self.calibration_windows)
        starts = np.asarray(dataset.sample_starts)[indices]
        errors = self._window_node_errors(dataset, starts)
        self.model.calibrate_from_errors(errors)
        self.calibration = self.model.get_calibration()

    def _window_node_errors(self, dataset, starts):
        """Mean one-step residual over the last ``residual_steps`` observations."""
        history_length = self.model.history_length
        groups, histories, targets = [], [], []
        for group, start in enumerate(np.asarray(starts, dtype=np.int64)):
            observation_end = int(start) + history_length - 1
            first_target = max(
                history_length, observation_end - self.residual_steps + 1
            )
            for target_index in range(first_target, observation_end + 1):
                if target_index < history_length:
                    continue
                groups.append(group)
                histories.append(
                    dataset.rawdata[target_index - history_length:target_index, :-1]
                )
                targets.append(dataset.rawdata[target_index, 1:-1])
        if not histories:
            return torch.zeros(
                len(starts), self.model.node_count,
                dtype=torch.float32, device=self.device,
            )
        errors = self._one_step_errors(histories, targets)
        grouped = torch.zeros(
            len(starts), self.model.node_count,
            dtype=errors.dtype, device=errors.device,
        )
        counts = torch.zeros(len(starts), 1, dtype=errors.dtype, device=errors.device)
        group_index = torch.as_tensor(groups, dtype=torch.long, device=errors.device)
        grouped.index_add_(0, group_index, errors)
        counts.index_add_(
            0, group_index, torch.ones(len(groups), 1, dtype=errors.dtype,
                                       device=errors.device),
        )
        return grouped / counts.clamp_min(1.0)

    def train(self):
        self.model.train()
        self.model.set_graph_temperature(
            self.model.annealed_temperature(self.step)
        )
        print('start at:', datetime.now())
        with tqdm(initial=self.step, total=self.train_num_steps) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.0
                teacher_probability = self.model.teacher_forcing_probability(self.step)
                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    data = batch[0].to(self.device)
                    history, target = self._split_windows(data)
                    training_target = (
                        target if random.random() < teacher_probability else None
                    )
                    predictions, graph = self.model(history, training_target)
                    loss = F.l1_loss(predictions, target)
                    if self.model.graph_l1:
                        loss = loss + self.model.graph_l1 * graph.mean()
                    (loss / self.gradient_accumulate_every).backward()
                    total_loss += loss.item() / self.gradient_accumulate_every

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                self.step += 1
                pbar.set_description(f'sagn loss: {total_loss:.6f}')
                pbar.update(1)
                self.model.set_graph_temperature(
                    self.model.annealed_temperature(self.step)
                )

                if (
                    self.step % self.snapshot_interval == 0
                    or self.step == self.train_num_steps
                ):
                    validation_loss = self._validation_loss()
                    self._consider_pool(validation_loss)
                    if validation_loss < self.best_validation:
                        self.best_validation = validation_loss
                        self.best_state = _clone_state(self.model.state_dict())
                        self.best_step = self.step
                        print(
                            f'best validation at step {self.step}: '
                            f'{validation_loss:.6f}'
                        )

        if self.best_state is None:
            self.best_state = _clone_state(self.model.state_dict())
            self.best_step = self.step
        self.model.load_state_dict(self.best_state)
        self.model.eval()
        self._fit_calibration()
        self.best_state = _clone_state(self.model.state_dict())
        if not any(entry['step'] == self.best_step for entry in self.pool):
            self._consider_pool(self.best_validation)
        self.pool.sort(key=lambda item: item['loss'])
        self.pool = self.pool[: self.pool_size]
        self.save('best')
        print('training complete')
        print('end at:', datetime.now())

    # ------------------------------------------------------------------
    # online stage helpers
    # ------------------------------------------------------------------
    def _ensure_online_optimizer(self):
        online = self.model.online_config
        if self._online_optimizer is None:
            lr = float(online.get('learning_rate', 1.0e-3))
            self._online_optimizer = Adam(
                self.model.parameters(), lr=lr, betas=(0.9, 0.96)
            )

    def _reset_online_optimizer(self):
        self._online_optimizer = None

    def _one_step_errors(self, histories, targets, chunk_size=4096):
        """One-step absolute errors, batched to bound memory during calibration."""
        history_array = np.asarray(histories, dtype=np.float32)
        target_array = np.asarray(targets, dtype=np.float32)
        errors = []
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for offset in range(0, len(history_array), chunk_size):
                    history_tensor = torch.as_tensor(
                        history_array[offset:offset + chunk_size],
                        dtype=torch.float32, device=self.device,
                    )
                    target_tensor = torch.as_tensor(
                        target_array[offset:offset + chunk_size],
                        dtype=torch.float32, device=self.device,
                    )
                    predictions = self.model.predict(history_tensor, steps=1)[:, 0]
                    errors.append((predictions - target_tensor).abs())
        finally:
            self.model.train(was_training)
        return torch.cat(errors, dim=0) if errors else torch.empty(
            0, self.model.node_count, dtype=torch.float32, device=self.device
        )

    def _online_update(self, histories, targets, epochs):
        self._ensure_online_optimizer()
        history_tensor = torch.as_tensor(
            np.stack(histories), dtype=torch.float32, device=self.device
        )
        target_tensor = torch.as_tensor(
            np.stack(targets), dtype=torch.float32, device=self.device
        )
        self.model.train()
        with torch.enable_grad():
            for _ in range(max(1, int(epochs))):
                predictions, _ = self.model(
                    history_tensor, target_tensor.unsqueeze(1), steps=1
                )
                loss = F.l1_loss(predictions, target_tensor.unsqueeze(1))
                self._online_optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                self._online_optimizer.step()
        self.model.eval()
        self.model.set_calibration(self.calibration)

    def _pool_loss(self, state, histories, targets):
        if self._eval_model is None:
            self._eval_model = copy.deepcopy(self.model).to(self.device)
        self._eval_model.load_state_dict(state)
        self._eval_model.eval()
        history_tensor = torch.as_tensor(
            np.stack(histories), dtype=torch.float32, device=self.device
        )
        target_tensor = torch.as_tensor(
            np.stack(targets), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            predictions = self._eval_model.predict(history_tensor, steps=1)[:, 0]
            return F.l1_loss(predictions, target_tensor).item()

    def _select_from_pool(self, validation_pairs):
        """Dynamic validation set and GNN pool rolling mechanism (Sec. 4.2.2)."""
        if not validation_pairs:
            return
        histories = [pair[0] for pair in validation_pairs]
        targets = [pair[1] for pair in validation_pairs]
        current_state = _clone_state(self.model.state_dict())
        current_loss = self._pool_loss(current_state, histories, targets)
        best_loss = current_loss
        best_index = None
        for index, entry in enumerate(self.pool):
            pool_loss = self._pool_loss(entry['state'], histories, targets)
            if pool_loss < best_loss:
                best_loss = pool_loss
                best_index = index

        event = {
            'selection': len(self._online_log) + 1,
            'step': self.step,
            'validation_size': len(validation_pairs),
            'current_loss': current_loss,
            'best_pool_loss': (
                self.pool[best_index]['loss'] if best_index is not None else None
            ),
            'best_validation_loss': best_loss,
            'selected': 'current' if best_index is None else 'pool',
        }
        online = self.model.online_config
        if best_index is None:
            if bool(online.get('pool_update', True)):
                if len(self.pool) < self.pool_size:
                    self.pool.append({
                        'state': current_state,
                        'loss': current_loss,
                        'step': self.step,
                    })
                else:
                    worst = max(range(len(self.pool)),
                                key=lambda i: self.pool[i]['loss'])
                    self.pool[worst] = {
                        'state': current_state,
                        'loss': current_loss,
                        'step': self.step,
                    }
                self.pool.sort(key=lambda item: item['loss'])
            event['action'] = 'keep_current'
        else:
            self.model.load_state_dict(self.pool[best_index]['state'])
            self.model.set_calibration(self.calibration)
            self.model.eval()
            self._reset_online_optimizer()
            event['action'] = 'roll_in_pool_model'
            event['pool_index'] = best_index
        self._online_log.append(event)

    def _new_observation_pairs(self, dataset, starts):
        """Observed (history, next metric) pairs for the newest timestamps."""
        history_length = self.model.history_length
        histories, targets = [], []
        for start in np.asarray(starts, dtype=np.int64):
            target_index = int(start) + history_length
            if target_index >= len(dataset.rawdata):
                continue
            histories.append(dataset.rawdata[target_index - history_length:target_index, :-1])
            targets.append(dataset.rawdata[target_index, 1:-1])
        return histories, targets

    def _window_aggregate(self, buffer, observation_end):
        """Anomaly magnitude from residuals observed no later than the window."""
        entries = [
            node_errors for target_index, node_errors in buffer
            if target_index <= observation_end
        ][-self.residual_steps:]
        if not entries:
            return 0.0
        node_errors = np.mean(np.stack(entries), axis=0)
        aggregate = self.model.aggregate_from_node_errors(
            torch.as_tensor(node_errors, dtype=torch.float32, device=self.device)
        )
        return float(aggregate.item())

    def _residual_score_for_window(self, buffer, observation_end,
                                   aggregate_scale=None):
        aggregate = self._window_aggregate(buffer, observation_end)
        if aggregate <= 0.0:
            return 0.0
        score = self.model.score_from_aggregate(
            torch.tensor(aggregate), aggregate_scale
        )
        return float(score.clamp(0.0, 1.0).item())

    # ------------------------------------------------------------------
    # online evaluation
    # ------------------------------------------------------------------
    @contextmanager
    def validation_prediction_context(self):
        """Run validation without leaking its online updates into test state."""
        state = _clone_state(self.model.state_dict())
        pool = copy.deepcopy(self.pool)
        calibration = copy.deepcopy(self.calibration)
        online_log = copy.deepcopy(self._online_log)
        training = self.model.training
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = (
            torch.cuda.get_rng_state(self.device)
            if self.device.type == 'cuda' else None
        )
        try:
            yield
        finally:
            self.model.load_state_dict(state)
            self.model.train(training)
            self.model.set_calibration(calibration)
            self.pool = pool
            self.calibration = calibration
            self._online_log = online_log
            self._online_optimizer = None
            self._eval_model = None
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state(cuda_rng, self.device)

    def restore(self, raw_dataloader, shape=None, coef=None, stepsize=None,
                sampling_steps=None):
        dataset = raw_dataloader.dataset
        if dataset.period not in {'validation', 'test'}:
            raise ValueError('SAGN evaluation requires a validation or test dataset.')
        if not isinstance(raw_dataloader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError('SAGN evaluation must be chronological.')
        if getattr(dataset, 'feature_scaling', None) != 'train_minmax':
            raise ValueError('SAGN online residuals require train_minmax scaling.')

        history_length = self.model.history_length
        online = self.model.online_config
        online_enabled = bool(online.get('enabled', True))
        online_epochs = max(1, int(online.get('epochs', 1)))
        window_batch = max(1, int(online.get('window_batch', 1)))
        warmup = max(0, int(online.get('warmup', 0)))
        validation_size = max(1, int(online.get('validation_size', 64)))
        baseline_windows = max(
            0, int(self.model.residual_config.get('baseline_windows', 0))
        )
        baseline_min = max(
            1, int(self.model.residual_config.get('baseline_min', 32))
        )
        aggregate_baseline = (
            deque(maxlen=baseline_windows) if baseline_windows else None
        )

        self.model.eval()
        self._online_log = []
        self._reset_online_optimizer()
        self._eval_model = None

        residual_buffer = deque(maxlen=self.residual_steps)
        last_residual_target = None
        observed = 0
        validation_pairs = []

        samples, probabilities, label_batches = [], [], []
        processed = 0
        with torch.no_grad():
            for batch_x, _, batch_label in raw_dataloader:
                batch_starts = np.asarray(
                    dataset.sample_starts[processed:processed + len(batch_x)]
                )
                processed += len(batch_x)
                for offset in range(0, len(batch_x), window_batch):
                    end = min(offset + window_batch, len(batch_x))
                    x = batch_x[offset:end].to(self.device)
                    starts = batch_starts[offset:end]
                    history = x[:, :history_length, :]

                    # Observed one-step residuals for the newest timestamps.
                    observation_ends = starts + history_length - 1
                    if last_residual_target is None:
                        first_target = max(
                            history_length, int(observation_ends[0])
                            - self.residual_steps + 1,
                        )
                    else:
                        first_target = last_residual_target + 1
                    new_histories, new_targets, new_indices = [], [], []
                    for target_index in range(first_target,
                                              int(observation_ends[-1]) + 1):
                        if target_index < history_length:
                            continue
                        new_indices.append(target_index)
                        new_histories.append(
                            dataset.rawdata[
                                target_index - history_length:target_index, :-1
                            ]
                        )
                        new_targets.append(dataset.rawdata[target_index, 1:-1])
                    if new_indices:
                        errors = self._one_step_errors(
                            new_histories, new_targets
                        ).cpu().numpy()
                        for target_index, node_error in zip(new_indices, errors):
                            residual_buffer.append((target_index, node_error))
                        last_residual_target = new_indices[-1]

                    # Autoregressive forecast for every window in this chunk.
                    predictions, _ = self.model(history, target=None,
                                                sample_graph=False)
                    aggregates = [
                        self._window_aggregate(residual_buffer, int(end_index))
                        for end_index in observation_ends
                    ]
                    baseline_scale = None
                    if (aggregate_baseline is not None
                            and len(aggregate_baseline) >= baseline_min):
                        positive = [value for value in aggregate_baseline
                                    if value > 0.0]
                        if positive:
                            baseline_scale = float(np.median(positive))
                    scores = [
                        float(self.model.score_from_aggregate(
                            torch.tensor(aggregate), baseline_scale
                        ).clamp(0.0, 1.0).item())
                        for aggregate in aggregates
                    ]
                    if aggregate_baseline is not None:
                        aggregate_baseline.extend(aggregates)
                    samples.append(predictions.cpu().numpy())
                    probabilities.append(np.asarray(scores, dtype=np.float64))
                    label_batches.append(
                        (batch_label[offset:end, -self.model.DeltaT:] != 0)
                        .any(dim=1).long().cpu().numpy()
                    )

                    if not online_enabled:
                        continue
                    histories, targets = self._new_observation_pairs(dataset, starts)
                    if not histories:
                        continue
                    if observed >= warmup:
                        self._online_update(histories, targets, online_epochs)
                    observed += len(histories)
                    if observed > warmup:
                        validation_pairs.extend(zip(histories, targets))
                        if len(validation_pairs) >= validation_size:
                            self._select_from_pool(validation_pairs[-validation_size:])
                            validation_pairs = []

        if not samples:
            raise ValueError('Cannot evaluate an empty forecast dataloader.')
        self.model.eval()
        self.model.set_calibration(self.calibration)
        self._write_online_log(dataset)
        return self._finish_forecast_report(
            dataset, samples, probabilities, label_batches
        )

    def _write_online_log(self, dataset):
        output_dir = Path(self.args.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f'{dataset.name}_sagn_online_{dataset.period}.json'
        payload = {
            'method': 'sagn',
            'split': dataset.period,
            'pool_size': len(self.pool),
            'selections': self._online_log,
        }
        with open(path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _finish_forecast_report(self, dataset, samples, probabilities, labels):
        pred_len = self.model.DeltaT
        checkpoint = str(self.results_folder / f'checkpoint-{self.milestone}.pt')
        report = write_detection_report(
            dataset=dataset,
            probabilities=np.concatenate(probabilities),
            labels=np.concatenate(labels),
            pred_len=pred_len,
            output_dir=self.args.save_dir,
            config=self.config['dataloader'],
            threshold=self.decision_threshold,
            checkpoint=checkpoint,
            checkpoint_step=self.step,
        )
        self.decision_threshold = report['threshold']
        self.last_detection_report = report
        metrics = report['metrics']
        if report['validation_threshold_selection'] is not None:
            print(
                'validation selected threshold:'
                f"{self.decision_threshold:.17g}"
            )
        print(
            f"{dataset.period} window-level pre:{metrics['precision']:.6f},"
            f"rec:{metrics['window_recall']:.6f},f1:{metrics['f1']:.6f}"
        )
        print(
            f"{dataset.period} event-level hits:"
            f"{metrics['matched_alarm_events']}/{metrics['eligible_onsets']},"
            f"lead_matched:{metrics['event_avg_lead_time_timestamps']:.6f},"
            'lead_all_eligible:'
            f"{metrics['event_avg_lead_time_all_eligible_onsets_timestamps']:.6f}"
            ' timestamps'
        )
        return np.concatenate(samples, axis=0)
