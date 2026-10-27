import math
import os
import sys
import random
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from einops import reduce
from scipy.special import softmax
from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from Models.interpretable_diffusion.model_utils import default, identity, extract

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl, on_restart=None):
    while True:
        if on_restart is not None:
            on_restart()
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, config, args, model, granger, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.granger = granger
        self.history_adaptation_enabled = getattr(model, 'history_adaptation_enabled', False)
        self.dataset = dataloader['dataset']
        if self.history_adaptation_enabled:
            if getattr(self.dataset, 'forecast_length', None) is None:
                raise ValueError('History drift adaptation requires future-interval labels and the forecast split protocol.')
            if not isinstance(dataloader['dataloader'].sampler, torch.utils.data.SequentialSampler):
                raise ValueError('History drift adaptation requires chronological training and evaluation loaders.')
            if self.dataset.var_num != self.model.feature_size:
                raise ValueError('Dataset feature count must match model.feature_size, including RPS and labels.')
            if not self.granger.joint_training:
                raise ValueError('History drift adaptation requires joint Granger training.')
        self.device = self.model.betas.device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'], model.reset_drift_state if self.history_adaptation_enabled else None)
        self.step = 0
        self.milestone = 0
        self.args, self.config = args, config
        self.logger = logger
        self.forecast_interval_labels = getattr(dataloader['dataset'], 'forecast_length', None) is not None
        self.decision_threshold = float(config['dataloader'].get('decision_threshold', 0.5))
        self.last_detection_report = None

        self.results_folder = Path(config['solver']['results_folder'] + f'_{model.seq_length}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)

        self.optG = Adam(filter(lambda p: p.requires_grad, self.granger.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.emaG = EMA(self.granger, beta=ema_decay, update_every=ema_update_every).to(self.device)
        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.optG
        self.schG = instantiate_from_config(sc_cfg)

        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

    def adaptation_protocol(self):
        return {'version': 1,
                'model_parameters': deepcopy(self.config['model']['params']),
                'drift_adaptation': self.config['model']['params'].get('drift_adaptation'),
                'feature_normalization': self.dataset.normalization_state(),
                'granger_joint_training': bool(self.granger.joint_training),
                'clip_denoised': self.model.clip_denoised}
        
    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'granger': self.granger.state_dict(),
            'emaG': self.emaG.state_dict(),
            'optG': self.optG.state_dict(),
        }
        if self.forecast_interval_labels:
            data['evaluation_protocol'] = {
                'label_scope': 'forecast_interval_any_fault',
                'train_proportion': self.config['dataloader']['params']['proportion'],
                'validation_proportion': self.config['dataloader']['params']['validation_proportion'],
                'stride': self.config['dataloader']['params']['DeltaT'],
                'pred_len': self.args.pred_len,
            }
        if self.history_adaptation_enabled:
            data['adaptation_protocol'] = self.adaptation_protocol()
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))
    
    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        if self.forecast_interval_labels:
            protocol = data.get('evaluation_protocol')
            if protocol is None or protocol.get('label_scope') != 'forecast_interval_any_fault':
                raise ValueError('This checkpoint does not use future-interval labels. Retrain with the 6/2/2 event config first.')
            params = self.config['dataloader']['params']
            expected = {
                'train_proportion': params['proportion'],
                'validation_proportion': params['validation_proportion'],
                'stride': params['DeltaT'], 'pred_len': self.args.pred_len,
            }
            if any(protocol.get(key) != value for key, value in expected.items()):
                raise ValueError('The checkpoint split, stride, or forecast horizon does not match this evaluation config.')
        if self.history_adaptation_enabled and data.get('adaptation_protocol') != self.adaptation_protocol():
            raise ValueError('The checkpoint does not match drift_v1 or train-only normalization. Retrain with the new config.')
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.granger.load_state_dict(data['granger'])
        self.step = data['step']
        self.optG.load_state_dict(data['optG'])
        self.emaG.load_state_dict(data['emaG'])
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0       
        DeltaT = self.config["model"]["params"]["DeltaT"]

        best_loss = float('inf')
        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            
            while step < self.train_num_steps:
                total_loss, total_lossCausal = 0., 0.
                for _ in range(self.gradient_accumulate_every):
                    data, labels = next(self.dl)
                    data = data.to(device)
                    labels = labels.to(device)
                    if self.forecast_interval_labels:
                        labels = labels[:, -DeltaT:]
                    # 训练过程预测目标下一个窗口的后 DeltaT 个timestamp数据, 模型预测数据并判断下一个窗口是否存在故障
                    loss, lossCausal = self.model(data=data, labels=labels, DeltaT=DeltaT, clip=self.args.clip, granger=self.granger) 
                    loss = loss / self.gradient_accumulate_every
                    lossCausal = lossCausal / self.gradient_accumulate_every
                    loss.backward()
                    lossCausal.backward()
                    total_loss += loss.item()
                    total_lossCausal += lossCausal.item()

                pbar.set_description(f'loss: {total_loss:.6f},lossCausal: {total_lossCausal:.6f}')

                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.sch.step(total_loss)
                self.opt.zero_grad()
                clip_grad_norm_(self.granger.parameters(), 1.0)
                self.optG.step()
                self.schG.step(total_lossCausal)
                self.optG.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()
                self.emaG.update()

                pbar.update(1)
                if total_loss < best_loss:
                    best_loss = total_loss
                    self.save("best")
                    print('best epoch at: ',step-1)

        print('training complete')


    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.forecast_interval_labels:
            return self._restore_forecast(raw_dataloader, coef, stepsize, sampling_steps)
        model_kwargs = {}
        DeltaT = self.args.pred_len
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, DeltaT, shape[1]-1])
        labels,judges = np.array([]), np.array([])

        self.granger.train()

        for idx, (x, t_m, label) in enumerate(raw_dataloader): # x 即 test 里取出的batch，预测过程 t_m all False

            x, t_m = x.to(self.device), t_m.to(self.device)    
            if DeltaT < x.shape[1]:
                source = x[:,:-DeltaT,:] 
                target = x[:,-DeltaT:,1:] 
                t_m = t_m[:,-DeltaT:,1:] 
            else:
                source = x[:-1,:,:]
                target = x[1:,:,1:]
                t_m = t_m[1:,:,1:]
            
            target_shape = list(target.shape)

            if sampling_steps == self.model.num_timesteps: # 条件rps
                sample, judge, total_lossCausal = self.ema.ema_model.sample_infill(granger=self.granger,DeltaT=DeltaT,shape=target_shape,history=source,clip=self.args.clip,target=target*t_m,labels=label,\
                                                                                partial_mask=t_m,model_kwargs=model_kwargs)
                label = (label != 0).any(dim=1).long().numpy()
                labels = np.concatenate([labels,label])
                judge = torch.argmax(judge, dim=1).cpu().numpy()
                judges = np.concatenate([judges,judge])
                
                clip_grad_norm_(self.granger.parameters(), 1.0)
                self.optG.step()
                self.schG.step(total_lossCausal)
                self.optG.zero_grad()
                self.emaG.update()
            else: # 无条件rps
                print(sample) 
            
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])


        precision = precision_score(labels, judges)  # Precision = TP / (TP + FP)
        recall = recall_score(labels, judges)      # Recall = TP / (TP + FN)
        f1 = f1_score(labels, judges) # 需要拿第一个窗口做预测
        print(f"pre:{precision:.6f},rec:{recall:.6f},f1:{f1:.6f}")

        return samples

    @contextmanager
    def validation_prediction_context(self):
        """Prevent validation's existing online Granger updates from changing test state."""
        state = {
            'granger': deepcopy(self.granger.state_dict()),
            'optimizer': deepcopy(self.optG.state_dict()),
            'ema': deepcopy(self.emaG.state_dict()),
            'scheduler': deepcopy(self.schG.state_dict()),
            'training': self.granger.training,
            'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(),
            'torch_rng': torch.get_rng_state(),
        }
        if self.device.type == 'cuda':
            state['cuda_rng'] = torch.cuda.get_rng_state(self.device)
        if self.history_adaptation_enabled:
            state['drift'] = self.model.model.drift_adapter.get_state()
            state['ema_drift'] = self.ema.ema_model.model.drift_adapter.get_state()
            state['ema_training'] = self.ema.ema_model.training
        try:
            yield
        finally:
            self.granger.load_state_dict(state['granger'])
            self.optG.load_state_dict(state['optimizer'])
            self.optG.zero_grad()
            self.emaG.load_state_dict(state['ema'])
            # This repository's scheduler recomputes warm-up increments in
            # load_state_dict. Restore its saved attributes exactly instead.
            self.schG.__dict__.update(state['scheduler'])
            self.granger.train(state['training'])
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            if 'cuda_rng' in state:
                torch.cuda.set_rng_state(state['cuda_rng'], self.device)
            if self.history_adaptation_enabled:
                self.model.model.drift_adapter.set_state(state['drift'])
                self.ema.ema_model.model.drift_adapter.set_state(state['ema_drift'])
                self.ema.ema_model.train(state['ema_training'])

    def _restore_forecast(self, raw_dataloader, coef, stepsize, sampling_steps):
        from Utils.detection_report import write_detection_report

        if sampling_steps != self.model.num_timesteps:
            raise NotImplementedError('Forecast evaluation requires sampling_steps to equal diffusion timesteps.')
        dataset = raw_dataloader.dataset
        if dataset.period not in {'validation', 'test'}:
            raise ValueError('Forecast evaluation requires a validation or test dataloader.')
        if not isinstance(raw_dataloader.sampler, torch.utils.data.SequentialSampler):
            raise ValueError('Forecast evaluation must use a chronological SequentialSampler.')
        if self.history_adaptation_enabled:
            return self._restore_history_forecast(raw_dataloader, coef, stepsize)
        pred_len = self.args.pred_len
        sample_batches, probability_batches, label_batches = [], [], []
        self.granger.train()
        model_kwargs = {'coef': coef, 'learning_rate': stepsize}
        for x, mask, label in raw_dataloader:
            x, mask = x.to(self.device), mask.to(self.device)
            source = x[:, :-pred_len, :]
            target = x[:, -pred_len:, 1:]
            target_mask = mask[:, -pred_len:, 1:]
            sample, logits, causal_loss = self.ema.ema_model.sample_infill(
                granger=self.granger, DeltaT=pred_len, shape=list(target.shape),
                history=source, clip=self.args.clip, target=target * target_mask,
                labels=None, partial_mask=target_mask, model_kwargs=model_kwargs,
            )
            probability_batches.append(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().numpy())
            label_batches.append((label[:, -pred_len:] != 0).any(dim=1).long().cpu().numpy())
            sample_batches.append(sample.detach().cpu().numpy())
            clip_grad_norm_(self.granger.parameters(), 1.0)
            self.optG.step()
            self.schG.step(causal_loss)
            self.optG.zero_grad()
            self.emaG.update()
        if not sample_batches:
            raise ValueError('Cannot evaluate an empty forecast dataloader.')

        return self._finish_forecast_report(dataset, sample_batches, probability_batches, label_batches)

    def _restore_history_forecast(self, raw_dataloader, coef, stepsize):
        import pandas as pd

        dataset = raw_dataloader.dataset
        diffusion = self.ema.ema_model
        diffusion.eval()
        diffusion.reset_drift_state()
        teacher = self.emaG.ema_model
        teacher.eval()
        self.granger.eval()
        pred_len = self.args.pred_len
        history_len = dataset.window - pred_len
        online_update = bool(self.config['dataloader'].get('online_granger_update', False))
        warmup = int(self.config['dataloader'].get('drift_warmup_windows', 128))
        first_start = int(dataset.sample_starts[0])
        warm_starts = range(max(0, first_start - warmup), first_start, dataset.DeltaT)
        with torch.no_grad():
            for offset in range(0, len(warm_starts), 64):
                starts = list(warm_starts[offset:offset + 64])
                history = np.stack([dataset.rawdata[start:start + history_len, :-1] for start in starts])
                diffusion.model.prepare_history(torch.as_tensor(history, device=self.device, dtype=torch.float32), teacher)
        sample_batches, probability_batches, label_batches, diagnostic_rows = [], [], [], []
        processed = 0
        for batch_x, _, batch_label in raw_dataloader:
            # Parameter updates require one observation at a time. The frozen default
            # batches diffusion while scanning drift memory in timestamp order.
            size = 1 if online_update else len(batch_x)
            for x, label in zip(batch_x.split(size), batch_label.split(size)):
                source = x[:, :-pred_len].to(self.device)
                with torch.no_grad():
                    context = diffusion.model.prepare_history(source, teacher)
                    diagnostics = deepcopy(diffusion.model.drift_adapter.last_diagnostics)
                    shape = [len(source), pred_len, source.shape[-1] - 1]
                    sample, logits, _ = diffusion.sample_infill(
                        granger=teacher, DeltaT=pred_len, shape=shape, history=source,
                        clip=self.args.clip, target=source.new_zeros(shape), labels=None,
                        partial_mask=torch.zeros(shape, dtype=torch.bool, device=self.device),
                        model_kwargs={'coef': coef, 'learning_rate': stepsize}, history_context=context)
                    probability_batches.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
                    sample_batches.append(sample.cpu().numpy())
                label_batches.append((label[:, -pred_len:] != 0).any(dim=1).long().numpy())
                for index, row in enumerate(diagnostics):
                    start = int(dataset.sample_starts[processed + index])
                    diagnostic_rows.append({'window_start': start, 'observation_end': start + history_len - 1, **row})
                processed += len(source)
                if online_update and diagnostics[0]['memory_updated']:
                    # Only the observed history supervises this auxiliary update.
                    self.optG.zero_grad()
                    self.granger.train()
                    loss, _ = self.granger(source[:, :, 1:], source[:, :, :1])
                    loss.backward()
                    clip_grad_norm_(self.granger.parameters(), 1.0)
                    self.optG.step()
                    self.schG.step(float(loss.detach()))
                    self.optG.zero_grad()
                    self.emaG.update()
                    teacher.eval()
                    self.granger.eval()
        if not sample_batches:
            raise ValueError('Cannot evaluate an empty forecast dataloader.')
        output_dir = Path(self.args.save_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostic_rows).to_csv(output_dir / f'{dataset.name}_drift_diagnostics_{dataset.period}.csv', index=False)
        return self._finish_forecast_report(dataset, sample_batches, probability_batches, label_batches)

    def _finish_forecast_report(self, dataset, sample_batches, probability_batches, label_batches):
        from Utils.detection_report import write_detection_report

        pred_len = self.args.pred_len
        checkpoint = str(self.results_folder / f'checkpoint-{self.milestone}.pt')
        report = write_detection_report(
            dataset=dataset,
            probabilities=np.concatenate(probability_batches),
            labels=np.concatenate(label_batches), pred_len=pred_len,
            output_dir=self.args.save_dir, config=self.config['dataloader'],
            threshold=self.decision_threshold, checkpoint=checkpoint, checkpoint_step=self.step,
        )
        self.decision_threshold = report['threshold']
        self.last_detection_report = report
        metrics = report['metrics']
        if report['validation_threshold_selection'] is not None:
            print(f"validation selected threshold:{self.decision_threshold:.17g}")
        print(f"{dataset.period} window-level pre:{metrics['precision']:.6f},rec:{metrics['window_recall']:.6f},f1:{metrics['f1']:.6f}")
        print(f"{dataset.period} event-level hits:{metrics['matched_alarm_events']}/{metrics['eligible_onsets']},"
              f"lead_matched:{metrics['event_avg_lead_time_timestamps']:.6f},"
              f"lead_all_eligible:{metrics['event_avg_lead_time_all_eligible_onsets_timestamps']:.6f} timestamps")
        return np.concatenate(sample_batches, axis=0)
