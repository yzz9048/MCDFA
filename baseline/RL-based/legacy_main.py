import numpy as np
import torch

import os
import argparse
from copy import deepcopy
from pathlib import Path
from engine.solver import Trainer
from Data.build_dataloader import *
from Models.interpretable_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config
from datetime import datetime
from Models.interpretable_diffusion.Granger import GrangerNet


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, required=True)

    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    # args for random

    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.') # action='store_true'不传参，出现即为 not default
    parser.add_argument('--cond', type=int, default=1, 
                        choices=[0, 1], help='RPS Condition or not.')
    parser.add_argument('--milestone', type=int, default=10)
    parser.add_argument('--split', choices=['validation', 'test'], default='test',
                        help='evaluation split; test first selects a threshold on validation')

    parser.add_argument('--pred_len', type=int, required=True,  help='the length of DeltaT')

    parser.add_argument('--clip', type=bool, default=True)

    args = parser.parse_args(argv)
    if args.config_file is None:
        config_name = args.name.casefold()
        config_path = Path(__file__).resolve().parent / 'config' / (
            'predict.yaml' if config_name == 'gaia' else f'predict-{config_name}.yaml'
        )
        if not config_path.is_file():
            parser.error(f'No config found for {args.name}; pass --config_file explicitly.')
        args.config_file = str(config_path)
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        
    config = load_yaml_config(args.config_file)
    history_adaptation = config['model']['params'].get('drift_adaptation', {}).get('enabled', False)
    if history_adaptation:
        if args.name.casefold() == 'gaia':
            raise ValueError('The history drift adaptation protocol excludes GAIA.')
        args.save_dir = os.path.join(args.output, args.name, 'drift_v1')
    params = config['dataloader']['params']
    forecast_protocol = params.get('validation_proportion', 0.0) > 0
    if forecast_protocol:
        if str(params['name']).casefold() != args.name.casefold():
            raise ValueError('--name must match the configured dataset.')
        if args.pred_len != config['model']['params']['DeltaT']:
            raise ValueError('--pred_len must match model.params.DeltaT (the forecast horizon).')
        if params['window'] != config['model']['params']['seq_length']:
            raise ValueError('Dataset window and model seq_length must match.')
        if not 0 < args.pred_len < params['window'] or params['window'] - args.pred_len <= 5:
            raise ValueError('The forecast horizon must leave more than 5 historical timestamps.')
    elif args.split != 'test':
        raise ValueError('This dataset has no validation split.')
    params['period'] = 'train' if args.train else args.split

    model = instantiate_from_config(config['model']).cuda()
    n_feat= config["model"]["params"]["feature_size"]
    seq_len=config["model"]["params"]["seq_length"]
    granger = GrangerNet(input_dim=n_feat-2, his_len=seq_len-args.pred_len,
                         joint_training=history_adaptation).cuda()

    dataloader_info = build_dataloader_cond(config, args) if args.cond == 1 else build_dataloader

    trainer = Trainer(config=config, args=args, model=model, granger=granger, dataloader=dataloader_info) 
    if args.train:
        print("start at:",datetime.now())
        trainer.train()
        print("end at:",datetime.now())
    elif args.cond == 1: # rps条件预测
        trainer.load("best")
        dataloader, dataset = dataloader_info['dataloader'], dataloader_info['dataset']
        coef = config['dataloader']['coefficient'] # 在Models/interpretable_diffusion/gaussian_diffusion.py用到
        stepsize = config['dataloader']['step_size']
        sampling_steps = config['dataloader']['sampling_steps']
        if forecast_protocol and args.split == 'test' and config['dataloader'].get('select_threshold_on_validation', True):
            validation_config = dict(config)
            validation_config['dataloader'] = deepcopy(config['dataloader'])
            validation_config['dataloader']['params']['period'] = 'validation'
            with trainer.validation_prediction_context():
                validation_info = build_dataloader_cond(validation_config, args)
                validation_dataset = validation_info['dataset']
                trainer.restore(
                    validation_info['dataloader'],
                    [validation_dataset.window, validation_dataset.var_num - 1],
                    coef, stepsize, sampling_steps,
                )
            del validation_info, validation_dataset
        samples = trainer.restore(dataloader, [dataset.window, dataset.var_num-1], coef, stepsize, sampling_steps)
        suffix = '_validation' if args.split == 'validation' else ''
        if history_adaptation:
            np.save(os.path.join(args.save_dir, f'{args.name}_prediction{suffix}.npy'), dataset.inverse_prediction(samples))
        # 采样的数据仅包含预测数据，ground truth的shape是【batch_size,window,feature_size】
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            # 反归一化，将生成数据恢复到原始范围，下同
            # samples = dataset.scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)
        np.save(os.path.join(args.save_dir, f'{args.name}_prediction_norm{suffix}.npy'), samples)
    else: # 仅依靠历史数据的条件预测
        trainer.load(args.milestone)

if __name__ == '__main__':
    main()

# train : python main.py --pred_len 10 --train --cond 1 --name maat 
# test : python main.py --pred_len 10 --cond 1 --name sn 

# train : python main.py --pred_len 10 --train --cond 1 --name GAIA --gpu 2
# test : python main.py --pred_len 10 --cond 1 --name GAIA 
