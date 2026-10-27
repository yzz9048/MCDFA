import os
import json
import torch
import numpy as np
import pandas as pd

from scipy import io
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from Utils.masking_utils import noise_mask


class CustomDataset(Dataset):
    def __init__(
        self, 
        name,
        data_root, 
        window=64, 
        DeltaT = 10,
        proportion=0.9, 
        validation_proportion=0.0,
        save2npy=True, 
        neg_one_to_one=True,
        seed=123,
        period='train',
        output_dir='./OUTPUT',
        forecast_length=None,
        predict_length=None,
        missing_ratio=None,
        style='separate', 
        distribution='geometric', 
        mean_mask_length=3,
        feature_scaling='legacy'
    ):
        super(CustomDataset, self).__init__()
        assert period in ['train', 'validation', 'test'], (
            'period must be train, validation, or test.'
        )
        self.forecast_length = forecast_length
        self.feature_scaling = feature_scaling
        if feature_scaling not in {'legacy', 'train_minmax'}:
            raise ValueError('feature_scaling must be legacy or train_minmax.')
        if feature_scaling != 'legacy' and forecast_length is None:
            raise ValueError('Train-only feature scaling requires the forecast split protocol.')
        self.validation_proportion = float(validation_proportion)
        if forecast_length is not None and period == 'train':
            assert predict_length is None and missing_ratio is None, (
                'predict_length and missing_ratio are inference-only parameters.'
            )
        elif period == 'train':
            assert ~(predict_length is not None or missing_ratio is not None), ''
        self.name, self.pred_len, self.missing_ratio = name, predict_length, missing_ratio
        self.style, self.distribution, self.mean_mask_length = style, distribution, mean_mask_length
        self.rawdata = self.read_data(data_root, self.name)
        self.raw_labels = self.rawdata[:, -1].copy()
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.DeltaT = DeltaT
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]
        self.sample_num_total = (self.rawdata.shape[0] - window)//DeltaT +1 # 可用的窗口数量
        self.save2npy = save2npy
        self.auto_norm = neg_one_to_one

        if forecast_length is not None:
            self._prepare_forecast_splits(data_root, proportion, seed)
            return
        if self.validation_proportion != 0 or period == 'validation':
            raise ValueError('A validation split requires forecast_length.')

        self.rawdata[:,:-1] = self.__normalize(self.rawdata[:,:-1]) 
        train, inference = self.__getsamples(self.rawdata, proportion, DeltaT, seed) # inference 即 test

        train_data = train[:,:,:-1]
        train_label = train[:,:,-1]
        inference_data = inference[:,:,:-1]
        inference_label = inference[:,:,-1]
        if period == 'train':
            self.samples = train_data  
            self.labels = train_label
        else:
            self.samples = inference_data  
            self.labels = inference_label
        if period == 'test':
            if predict_length is not None:
                masks = np.ones(self.samples.shape)
                masks[:, -predict_length:, :] = 0
                self.masking = masks.astype(bool)
            else:
                raise NotImplementedError()
        self.sample_num = self.samples.shape[0]

    def _prepare_forecast_splits(self, data_root, proportion, seed):
        """Split forecast targets on the raw timeline, as in CDFA-bk."""
        for name, value in (('window', self.window), ('DeltaT', self.DeltaT)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f'{name} must be a positive integer, got {value!r}.')
        horizon = self.forecast_length
        if isinstance(horizon, (bool, np.bool_)) or not isinstance(horizon, (int, np.integer)) or not 0 < horizon < self.window:
            raise ValueError('forecast_length must satisfy 0 < forecast_length < window.')
        if self.pred_len is not None and self.pred_len != horizon:
            raise ValueError('predict_length and forecast_length must match.')
        if self.window > self.len:
            raise ValueError('window cannot exceed the number of timestamps.')
        if not 0 < proportion < 1 or not 0 < self.validation_proportion < 1 - proportion:
            raise ValueError('Train and validation proportions must leave a non-empty test segment.')
        if self.period != 'train' and self.pred_len is None:
            raise ValueError('Validation/test datasets require predict_length.')

        self.raw_timestamps = pd.read_csv(data_root, usecols=[0]).iloc[:, 0].to_numpy()
        self.split_timestamp = int(np.ceil(self.len * proportion))
        self.validation_timestamp = int(np.ceil(self.len * (proportion + self.validation_proportion)))
        self.all_sample_starts = np.arange(0, self.len - self.window + 1, self.DeltaT, dtype=np.int64)
        self.sample_num_total = len(self.all_sample_starts)
        target_starts = self.all_sample_starts + self.window - horizon
        window_ends = self.all_sample_starts + self.window
        split_masks = {
            'train': window_ends <= self.split_timestamp,
            'validation': (target_starts >= self.split_timestamp) & (window_ends <= self.validation_timestamp),
            'test': target_starts >= self.validation_timestamp,
        }
        self.split_window_counts = {name: int(mask.sum()) for name, mask in split_masks.items()}
        self.boundary_crossing_windows_excluded = self.sample_num_total - sum(self.split_window_counts.values())
        if any(count == 0 for count in self.split_window_counts.values()):
            raise ValueError('The timeline is too short to form non-empty train, validation, and test forecast splits.')

        if self.feature_scaling == 'train_minmax':
            self.rawdata = self.rawdata.astype(np.float64, copy=False)
            if not np.isfinite(self.rawdata).all():
                raise ValueError('Features and labels must contain only finite values.')
            self.scaler = MinMaxScaler().fit(self.rawdata[:self.split_timestamp, :-1])
            if self.save2npy:
                with open(os.path.join(self.dir, f'{self.name}_normalization.json'), 'w', encoding='utf-8') as handle:
                    json.dump(self.normalization_state(), handle, indent=2)
        self.rawdata[:, :-1] = self.__normalize(self.rawdata[:, :-1])
        windows = np.stack([self.rawdata[start:start + self.window] for start in self.all_sample_starts])
        for split_name, mask in split_masks.items():
            split = windows[mask]
            features = split[:, :, :-1]
            if self.save2npy:
                np.save(os.path.join(self.dir, f'{self.name}_ground_truth_{self.window}_{split_name}.npy'), self.unnormalize(features))
                normalized = unnormalize_to_zero_to_one(features) if self.auto_norm else features
                np.save(os.path.join(self.dir, f'{self.name}_norm_truth_{self.window}_{split_name}.npy'), normalized)
            if split_name == self.period:
                self.samples = features
                self.labels = split[:, :, -1]
                self.sample_starts = self.all_sample_starts[mask]
        self.sample_num = len(self.samples)
        if self.period != 'train':
            self.masking = np.ones(self.samples.shape, dtype=bool)
            self.masking[:, -horizon:, :] = False

    def __getsamples(self, data, proportion, DeltaT, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        i = 0
        for w in range(self.window, data.shape[0] - self.window + 1, DeltaT): # 按窗口切分self.data
            end= w
            start = w - self.window
            x[i, :, :] = data[start:end, :]
            i += 1

        train_data, test_data = self.divide(x, proportion, seed)

        train_data_raw = train_data[:,:,:-1]
        test_data_raw = test_data[:,:,:-1]
        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"), self.unnormalize(test_data_raw))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"), self.unnormalize(train_data_raw))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), unnormalize_to_zero_to_one(test_data_raw))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), unnormalize_to_zero_to_one(train_data_raw))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data_raw)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data_raw)

        return train_data, test_data

    def normalize(self, sq):
        if self.feature_scaling == 'train_minmax':
            shape = np.asarray(sq).shape
            if shape[-1] != self.var_num - 1:
                raise ValueError('Feature normalization expects RPS and metrics, without labels.')
            data = self.scaler.transform(np.asarray(sq).reshape(-1, shape[-1]))
            if self.auto_norm:
                data = normalize_to_neg_one_to_one(data)
            return data.reshape(shape)
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num-1))
        return d.reshape(-1, self.window, self.var_num-1)
    
    def __normalize(self, rawdata):
        if self.feature_scaling == 'train_minmax':
            data = self.scaler.transform(rawdata)
            return normalize_to_neg_one_to_one(data) if self.auto_norm else data
        data = rawdata[:,0].reshape(-1, 1) # 调整到 【0，1】
        data = np.concatenate([data,rawdata[:,1:]], axis=1)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data) #调整到 【-1，1】
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        if self.feature_scaling == 'train_minmax':
            return self.scaler.inverse_transform(data)
        x = data[:,0].reshape(-1, 1)
        x = np.concatenate([x,data[:,1:]], axis=1)
        return x

    def normalization_state(self):
        state = {'feature_scaling': self.feature_scaling, 'neg_one_to_one': self.auto_norm}
        if self.feature_scaling == 'train_minmax':
            state.update({'data_min': self.scaler.data_min_.tolist(),
                          'data_max': self.scaler.data_max_.tolist(),
                          'scale': self.scaler.scale_.tolist(), 'offset': self.scaler.min_.tolist()})
        return state

    def inverse_prediction(self, prediction):
        """Invert metric-only forecasts (RPS is observed, never forecast)."""
        data = unnormalize_to_zero_to_one(prediction) if self.auto_norm else np.asarray(prediction)
        if self.feature_scaling == 'train_minmax':
            return (data - self.scaler.min_[1:]) / self.scaler.scale_[1:]
        return data
    
    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        # id_rdm = np.random.permutation(size)
        id_rdm = np.arange(size)
        regular_train_id = id_rdm[:regular_train_num] # 前窗口数*proportion为train，后为test
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        # if name == 'drift':
            # df.drop(df.columns[0:2], axis=1, inplace=True) # 删除前两列
        df.drop(df.columns[0], axis=1, inplace=True) # 删除第一列
        data = df.values
        return data
    
    def mask_data(self, seed=2023):
        masks = np.ones_like(self.samples)
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        for idx in range(self.samples.shape[0]):
            x = self.samples[idx, :, :]  # (seq_length, feat_dim) array
            mask = noise_mask(x, self.missing_ratio, self.mean_mask_length, self.style,
                              self.distribution)  # (seq_length, feat_dim) boolean array
            masks[idx, :, :] = mask

        if self.save2npy:
            np.save(os.path.join(self.dir, f"{self.name}_masking_{self.window}.npy"), masks)

        # Restore RNG.
        np.random.set_state(st0)
        return masks.astype(bool)

    def __getitem__(self, ind):
        if self.period in ['validation', 'test']:
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
            labels = self.labels[ind, :]
            return torch.from_numpy(x).float(), torch.from_numpy(m), torch.from_numpy(labels)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        labels = self.labels[ind, :]
        return torch.from_numpy(x).float(), torch.from_numpy(labels)

    def __len__(self):
        return self.sample_num
    

class fMRIDataset(CustomDataset):
    def __init__(
        self, 
        proportion=1., 
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler
