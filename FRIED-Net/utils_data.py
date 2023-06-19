import h5py
from os.path import join as pjoin
import numpy as np


def read_y_n(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('y_n_noisy')
        label = hf.get('y_n')
        return np.array(data), np.array(label)


def read_t_k(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('t_k')
        return np.squeeze(np.array(data))


def read_a_k(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('a_k')
        return np.squeeze(np.array(data))


