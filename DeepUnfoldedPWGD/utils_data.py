from scipy.linalg import toeplitz
from scipy import signal
import h5py
import scipy.io as sio
from os.path import join as pjoin
import numpy as np


def toeplitz_S(s_m, J):
    """
    NOT FOR BACKPROP
    Args:
        s_m: batched matrix of s_m or s_m
        J: the number of rows - 1
    Returns:
        Batched Toeplitz matrix or a single Toeplitz matrix
    """

    if s_m.ndim == 1:
        S_toe = toeplitz(s_m[J:], s_m[J::-1])
    elif s_m.ndim == 2:
        num_data = s_m.shape[0]
        P = s_m.shape[-1] - 1
        S_toe = np.empty([num_data, P - J + 1, J + 1], dtype=s_m.dtype)
        for i in range(num_data):
            S_toe[i, :] = toeplitz(s_m[i, J:], s_m[i, J::-1])

    return S_toe


def cat_cp(S):
    #Concatenate real and imaginary parts to form a new matrix
    S_r = np.concatenate((np.real(S), np.imag(S)), -1)
    return S_r


def split_cp(x):
    n = x.shape[-1]
    nh = int(n / 2)
    xR, xI = x[..., :nh], x[..., nh:]
    try:
        return xR.squeeze(-1), xI.squeeze(-1)
    except ValueError:
        return xR, xI


def read_y_n(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('y_n_noisy')
        label = hf.get('y_n')
        return np.array(data), np.array(label)


def read_s_m(data_dir, data_file, mat_file):
    if mat_file:
        y_n_noisy, y_n = read_y_n(data_dir, data_file)
        c_mn = read_c_mn(data_dir, mat_file)

        s_m_noisy = y_n_noisy @ c_mn.T
        s_m = y_n @ c_mn.T

        return s_m_noisy, s_m

    else:
        file_path = pjoin(data_dir, data_file)
        with h5py.File(file_path, 'r') as hf:
            data = hf.get('s_m_noisy')          # data original shape (num_data, 2*(P+1))
            label = hf.get('s_m')               # data original shape (num_data, 2*(P+1))

            data_real, data_imag = split_cp(data)
            data = data_real + 1j * data_imag

            label_real, label_imag = split_cp(label)
            label = label_real + 1j * label_imag

            return np.array(data), np.array(label)


def read_t_k(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('t_k')
        return np.array(data)


def read_h(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('h')              # data original shape (num_data, 2*(K+1)) [h_real, h_imag]
        hR, hI = split_cp(data)
        h = hR + 1j * hI
        return np.array(h)


def read_c_mn(data_dir, data_file):
    mat_fpath = pjoin(data_dir, data_file)
    contents = sio.loadmat(mat_fpath)
    c_mn = contents['c_mn']
    return c_mn


def read_a_k(data_dir, file):
    file_path = pjoin(data_dir, file)
    with h5py.File(file_path, 'r') as hf:
        data = hf.get('a_k')
        return np.array(data)

