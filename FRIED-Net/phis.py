import numpy as np
import scipy.io as sio


def eMOMS(resolution, P):
    P = int(P)

    t_phi = np.arange(0, P+1, 1 / resolution)
    lambda_0 = 2*np.pi/(P+1)
    i_vec = np.arange(1, P/2+1)
    tmp = 2*np.cos(lambda_0 * (t_phi[:, np.newaxis] - P / 2) * i_vec)
    phi = np.sum(tmp, -1) + 1

    return t_phi, phi/(P+1)


def ESpline(resolution, P, L):
    P = int(P)
    L = float(L)

    L = L*(P+1)
    m = np.arange(P+1)

    alpha_0 = - 1j * np.pi/ L * P
    alpha_vec = alpha_0 + 2j * np.pi / L * m

    len = (P + 1) * resolution
    N = 2 ** np.ceil(np.log2(len))
    w = (2 * np.pi / N) * np.arange(-N/2, N/2) * resolution

    X, Y = np.meshgrid(alpha_vec, 1j * w)
    num = 1 - np.exp(X - Y)
    denum = Y - X
    indet_idx = (num == 0) & (denum == 0)
    num[indet_idx] = 1
    denum[indet_idx] = 1
    phi_w = np.prod(num/denum, -1)

    phi = np.real(np.fft.ifft(np.fft.fftshift(phi_w)))
    phi = phi[:len]
    t_phi = np.arange(0, P+1, 1 / resolution)

    return t_phi, phi/np.max(phi)


def load_phi(mat_fpath):
    contents = sio.loadmat(mat_fpath)
    phi = contents['phi']
    t_phi = contents['t_phi']

    return t_phi, phi


def plot_phi_ReLU(c_ReLU, shift, init, resolution, T=1):
    c_ReLU = np.array(c_ReLU)
    shift = np.array(shift)
    init = np.array(init)

    T_int = 1/resolution

    y = np.insert(np.cumsum(np.cumsum(c_ReLU[:-1])), 0, 0)*T_int + init
    t = -shift * T

    return t, y


def plot_phi_tri(c_tri, shift, resolution, T=1):
    c_tri = np.array(c_tri)
    shift = np.array(shift)

    T_int = 1/resolution

    y = c_tri * T_int
    t = -shift[1:-1] * T

    return t, y