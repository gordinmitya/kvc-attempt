import numpy as np
import h5py
from tqdm import tqdm


def load_train_data_h5(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for group_key in tqdm(f.keys(), desc=f'Loading {file_path}', leave=True):
            group_value = {}
            for inner_key in f[group_key].keys():
                group_value[inner_key] = np.array(f[group_key][inner_key])
            data[group_key] = group_value
    return data


def load_test_data_h5(file_path):
    data = {}
    with h5py.File(file_path, 'r') as f:
        for key in tqdm(f.keys(), desc=f'Loading {file_path}', leave=True):
            data[key] = np.array(f[key])
    return data


def load_train_data(scenario):
    return load_train_data_h5(f'{scenario}/{scenario}_dev_set.h5')


def load_test_data(scenario):
    return load_test_data_h5(f'{scenario}/{scenario}_test_sessions.h5')
