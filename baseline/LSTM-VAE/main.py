import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from Data.build_dataloader import build_dataloader
from Utils.io_utils import load_yaml_config, seed_everything, instantiate_from_config
from engine.encdec_solver import EncDecTrainer


def resolve_encdec_config(name):
    """Resolve the LSTM encoder-decoder config for a dataset name."""
    key = str(name).casefold()
    filename = 'encdec-gaia.yaml' if key == 'gaia' else f'encdec-{key}.yaml'
    path = Path(__file__).resolve().parent / 'config' / filename
    if not path.is_file():
        raise FileNotFoundError(
            f'No EncDec-AD config found for {name!r}; expected {path}. '
            'Pass --config_file explicitly.'
        )
    return str(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='LSTM encoder-decoder anomaly detection'
    )
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--config_file', type=str, default=None,
                        help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT',
                        help='directory to save the results')
    parser.add_argument('--seed', type=int, default=12345,
                        help='seed for initializing training')
    parser.add_argument('--gpu', type=int, default=2,
                        help='GPU id to use; falls back to CPU when unavailable')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Train or Test.')
    parser.add_argument('--split', choices=['validation', 'test'], default='test',
                        help='evaluation split; test reuses the validation threshold')
    parser.add_argument('--checkpoint', type=str, default='best',
                        help='checkpoint name for evaluation')

    args = parser.parse_args(argv)
    if args.config_file is None:
        args.config_file = resolve_encdec_config(args.name)
    args.save_dir = os.path.join(args.output, args.name)
    return args


def select_device(args):
    if args.gpu is not None and torch.cuda.is_available():
        count = torch.cuda.device_count()
        index = int(args.gpu)
        if 0 <= index < count:
            torch.cuda.set_device(index)
            return torch.device(f'cuda:{index}')
        print(f'Warning: gpu {index} is unavailable; using cuda:0.')
        torch.cuda.set_device(0)
        return torch.device('cuda:0')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


def _check_feature_size(model, dataset):
    if getattr(dataset, 'task', None) != 'reconstruction':
        raise ValueError('The LSTM encoder-decoder requires reconstruction datasets.')
    if int(model.feature_size) != int(dataset.samples.shape[-1]):
        raise ValueError(
            f'model.feature_size={model.feature_size} does not match '
            f'dataset features={dataset.samples.shape[-1]}.'
        )


def _warn_about_labels(dataset, split):
    if not np.any(dataset.window_labels):
        print(
            f'Warning: the {split} split contains no labelled anomalous windows; '
            'recall and F-beta cannot be estimated on it.'
        )


def _build_split(config, args, period):
    split_config = deepcopy(config)
    split_config['dataloader']['params']['period'] = period
    split_config['dataloader']['params']['normal_only'] = False
    return build_dataloader(split_config, args)


def run_encdec(args):
    """Normal-only LSTM encoder-decoder reconstruction path."""
    config = load_yaml_config(args.config_file)
    params = config['dataloader']['params']
    if params.get('task') != 'reconstruction':
        raise ValueError(
            'EncDec-AD requires dataloader.params.task: reconstruction.'
        )

    device = select_device(args)
    os.makedirs(args.save_dir, exist_ok=True)
    model = instantiate_from_config(config['model']).to(device)

    if args.train:
        params['period'] = 'train'
        params['normal_only'] = bool(params.get('normal_only', True))
        train_info = build_dataloader(config, args)
        _check_feature_size(model, train_info['dataset'])
        validation_info = _build_split(config, args, 'validation')
        _check_feature_size(model, validation_info['dataset'])
        _warn_about_labels(validation_info['dataset'], 'validation')
        trainer = EncDecTrainer(
            config, args, model, train_info, validation_info=validation_info
        )
        print('start at:', datetime.now())
        trainer.train()
        print('end at:', datetime.now())
        return trainer

    validation_info = _build_split(config, args, 'validation')
    _check_feature_size(model, validation_info['dataset'])
    _warn_about_labels(validation_info['dataset'], 'validation')
    test_info = None
    if args.split == 'test':
        test_info = _build_split(config, args, 'test')
        _check_feature_size(model, test_info['dataset'])
        _warn_about_labels(test_info['dataset'], 'test')
    trainer = EncDecTrainer(
        config, args, model, dataloader=None, validation_info=validation_info
    )
    trainer.evaluate(validation_info=validation_info, test_info=test_info)
    return trainer


def main():
    args = parse_args()
    if args.seed is not None:
        seed_everything(args.seed)
    run_encdec(args)


if __name__ == '__main__':
    main()

# Train          : python main.py --name sn --train
# Test           : python main.py --name sn
# Validation only: python main.py --name sn --split validation
