from copy import deepcopy

import torch

from Utils.io_utils import instantiate_from_config


def build_dataloader(config, args=None):
    """Build a chronological loader for the configured dataset."""
    batch_size = config['dataloader']['batch_size']
    dataset_config = deepcopy(config['dataloader'])
    params = dataset_config['params']
    if args is not None:
        params['output_dir'] = args.save_dir
    dataset = instantiate_from_config(dataset_config)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=None,
        drop_last=False,
    )
    return {
        'dataloader': dataloader,
        'dataset': dataset,
    }
