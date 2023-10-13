import h5py
import numpy as np
from tqdm import tqdm
import os

assert os.path.exists('/.dockerenv'), "Must be running in docker container"


"""
docker run --rm -it -v "$(pwd):/app" python /bin/bash
pip install tqdm h5py numpy
python /app/convert_2_h5.py
"""


def convert_dev(input_file):
    data = np.load(input_file, allow_pickle=True).item()

    output_file = input_file.replace('.npy', '.h5')
    with h5py.File(output_file, 'w') as f:
        for group_key, group_value in tqdm(data.items(), desc=output_file):
            grp = f.create_group(group_key)
            for inner_key, array_value in group_value.items():
                grp.create_dataset(inner_key, data=array_value)


def convert_test(input_file):
    data = np.load(input_file, allow_pickle=True).item()

    output_file = input_file.replace('.npy', '.h5')
    with h5py.File(output_file, 'w') as f:
        for key, value in tqdm(data.items(), desc=output_file):
            f.create_dataset(key, data=value)


for t in ['desktop', 'mobile']:
    convert_dev(f'/app/{t}/{t}_dev_set.npy')
    convert_test(f'/app/{t}/{t}_test_sessions.npy')
