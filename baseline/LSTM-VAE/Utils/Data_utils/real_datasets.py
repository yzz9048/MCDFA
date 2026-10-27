import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

from Utils.normalization import (
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
)


class CustomDataset(Dataset):
    """Chronological reconstruction windows for the LSTM encoder-decoder.

    Windows are assigned to train/validation/test only when the whole window
    lies inside that segment of the raw timeline.  Training windows that
    contain any labelled fault are dropped, so the model sees normal behaviour
    only.  Scaling is fitted on the training segment and reused everywhere.
    """

    def __init__(
        self,
        name,
        data_root,
        window=64,
        DeltaT=10,
        proportion=0.9,
        validation_proportion=0.0,
        save2npy=True,
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        feature_scaling='train_minmax',
        task='reconstruction',
        normal_only=False,
    ):
        super().__init__()
        if period not in ['train', 'validation', 'test']:
            raise ValueError('period must be train, validation, or test.')
        if task != 'reconstruction':
            raise ValueError('CustomDataset supports task: reconstruction only.')
        if feature_scaling != 'train_minmax':
            raise ValueError('feature_scaling must be train_minmax.')
        if normal_only and period != 'train':
            raise ValueError('normal_only is only meaningful for the training split.')

        self.task = task
        self.name = name
        self.window = window
        self.period = period
        self.DeltaT = DeltaT
        self.validation_proportion = float(validation_proportion)
        self.feature_scaling = feature_scaling
        self.normal_only = bool(normal_only)
        self.seed = seed
        self.save2npy = bool(save2npy)
        self.auto_norm = bool(neg_one_to_one)
        self.dir = os.path.join(output_dir, 'samples')
        if self.save2npy:
            os.makedirs(self.dir, exist_ok=True)

        self.rawdata = self.read_data(data_root)
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self._prepare_reconstruction_splits(data_root, proportion)

    def _prepare_reconstruction_splits(self, data_root, proportion):
        for name, value in (('window', self.window), ('DeltaT', self.DeltaT)):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value <= 0
            ):
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        if self.window > self.len:
            raise ValueError('window cannot exceed the number of timestamps.')
        if not 0 < proportion < 1:
            raise ValueError('proportion must be in (0, 1).')
        if not 0 < self.validation_proportion < 1 - proportion:
            raise ValueError(
                'validation_proportion must be positive and leave a non-empty '
                'test segment.'
            )

        self.raw_timestamps = pd.read_csv(
            data_root, usecols=[0]
        ).iloc[:, 0].to_numpy()
        if len(self.raw_timestamps) != self.len:
            raise ValueError('Timestamp and feature rows must have the same length.')

        self.split_timestamp = int(np.ceil(self.len * proportion))
        self.validation_timestamp = int(
            np.ceil(self.len * (proportion + self.validation_proportion))
        )
        self.all_sample_starts = np.arange(
            0, self.len - self.window + 1, self.DeltaT, dtype=np.int64
        )
        window_ends = self.all_sample_starts + self.window
        split_masks = {
            'train': window_ends <= self.split_timestamp,
            'validation': (
                (self.all_sample_starts >= self.split_timestamp)
                & (window_ends <= self.validation_timestamp)
            ),
            'test': self.all_sample_starts >= self.validation_timestamp,
        }
        self.split_window_counts = {
            split: int(mask.sum()) for split, mask in split_masks.items()
        }
        self.boundary_crossing_windows_excluded = (
            len(self.all_sample_starts) - sum(self.split_window_counts.values())
        )
        if any(count == 0 for count in self.split_window_counts.values()):
            raise ValueError(
                'The timeline is too short to form non-empty reconstruction splits.'
            )

        self.rawdata = self.rawdata.astype(np.float64, copy=False)
        if not np.isfinite(self.rawdata).all():
            raise ValueError('Features and labels must contain only finite values.')

        self.scaler = MinMaxScaler().fit(
            self.rawdata[:self.split_timestamp, :-1]
        )
        if self.save2npy:
            with open(
                os.path.join(self.dir, f'{self.name}_normalization.json'),
                'w',
                encoding='utf-8',
            ) as handle:
                json.dump(self.normalization_state(), handle, indent=2)

        features = self.scaler.transform(self.rawdata[:, :-1])
        if self.auto_norm:
            features = normalize_to_neg_one_to_one(features)

        windows = np.stack(
            [features[start:start + self.window] for start in self.all_sample_starts]
        )
        point_labels = np.stack(
            [
                self.rawdata[start:start + self.window, -1]
                for start in self.all_sample_starts
            ]
        )
        window_labels = (point_labels != 0).any(axis=1).astype(np.int64)
        self.all_window_labels = window_labels

        if self.save2npy:
            for split_name, mask in split_masks.items():
                split_windows = windows[mask]
                np.save(
                    os.path.join(
                        self.dir,
                        f'{self.name}_reconstruction_truth_'
                        f'{self.window}_{split_name}.npy',
                    ),
                    self.unnormalize(split_windows),
                )
                np.save(
                    os.path.join(
                        self.dir,
                        f'{self.name}_norm_truth_{self.window}_{split_name}.npy',
                    ),
                    (
                        unnormalize_to_zero_to_one(split_windows)
                        if self.auto_norm
                        else split_windows
                    ),
                )

        selected = split_masks[self.period]
        if self.period == 'train' and self.normal_only:
            selected = selected & (window_labels == 0)
            if not selected.any():
                raise ValueError('No fault-free training windows are available.')
        self.samples = windows[selected]
        self.labels = point_labels[selected]
        self.window_labels = window_labels[selected]
        self.sample_starts = self.all_sample_starts[selected]
        self.sample_num = len(self.samples)

    def unnormalize(self, values):
        values = np.asarray(values, dtype=np.float64)
        shape = values.shape
        data = values.reshape(-1, shape[-1])
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        data = self.scaler.inverse_transform(data)
        return data.reshape(shape)

    def normalization_state(self):
        return {
            'feature_scaling': self.feature_scaling,
            'neg_one_to_one': self.auto_norm,
            'data_min': self.scaler.data_min_.tolist(),
            'data_max': self.scaler.data_max_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'offset': self.scaler.min_.tolist(),
        }

    @staticmethod
    def read_data(filepath):
        df = pd.read_csv(filepath, header=0)
        df.drop(df.columns[0], axis=1, inplace=True)
        return df.values

    def __getitem__(self, ind):
        x = self.samples[ind, :, :]
        labels = self.labels[ind, :]
        return torch.from_numpy(x).float(), torch.from_numpy(labels)

    def __len__(self):
        return self.sample_num
