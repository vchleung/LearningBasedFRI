import h5py
from os.path import join as pjoin
import numpy as np
import scipy.io as sio


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
        return np.array(data)


def read_a_k(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('a_k')
        return np.array(data)


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def read_y_n_cai(data_dir, file, N, raw=False, normalise=False, remove_bias=True, remove_mean=False):
    contents = sio.loadmat(pjoin(data_dir, file))

    fmean_roi = contents['obj']['timeSeriesArrayHash'][0][0][0][0]['value'][0][0][0][0]['valueMatrix']
    fmean_comp = fmean_roi
    fmean_neuropil = contents['obj']['timeSeriesArrayHash'][0][0][0][0]['value'][0][1][0][0]['valueMatrix']
    if not raw:
        fmean_comp = fmean_comp - 0.7 * fmean_neuropil
    y_n_noisy = fmean_comp.squeeze()
    y_n_noisy = y_n_noisy - y_n_noisy.min()

    if normalise:
        y_n_noisy = y_n_noisy / max(y_n_noisy)
        print("normalised")

    y_n_noisy = rolling_window(y_n_noisy, N)

    if remove_bias:
        y_n_noisy = y_n_noisy - y_n_noisy.min(axis=1, keepdims=True)
    if remove_mean:
        y_n_noisy = y_n_noisy - y_n_noisy.mean(axis=1, keepdims=True)

    return y_n_noisy