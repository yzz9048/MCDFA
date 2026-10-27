from copy import deepcopy

import torch
from Utils.io_utils import instantiate_from_config

def build_dataloader(config, args=None):
    return


def build_dataloader_cond(config, args=None):
    batch_size = config['dataloader']['batch_size']
    dataset_config = deepcopy(config['dataloader'])
    params = dataset_config['params']
    params['output_dir'] = args.save_dir
    period = params.get('period', 'train')
    if params.get('validation_proportion', 0.0) > 0:
        params['forecast_length'] = args.pred_len
        params['predict_length'] = None if period == 'train' else args.pred_len
        if period != 'train':
            batch_size = config['dataloader'].get('prediction_batch_size', batch_size)
    else:
        params['predict_length'] = args.pred_len
    dataset = instantiate_from_config(dataset_config)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=None,
                                             drop_last=False)

    dataload_info = {
        'dataloader': dataloader,
        'dataset': dataset
    }

    return dataload_info


if __name__ == '__main__':
    pass

