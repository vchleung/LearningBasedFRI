import numpy as np
import utils_data


def prony(s_m_est, K):
    ''' Implement Prony's method on the batched sum of exponentials (size = batch_size, P+1), Amplitudes are ignored in this routine'''

    num_data = len(s_m_est)
    S_K = utils_data.toeplitz_S(s_m_est, K)

    U, S, Vh = np.linalg.svd(S_K, full_matrices=False)
    h_est = np.conj(Vh[..., -1, :])

    u_k_est = np.zeros((num_data, K), dtype=s_m_est.dtype)
    for i in range(num_data):
        u_k_est[i] = np.roots(h_est[i])

    return u_k_est, h_est


def freq_to_time(u_k_est, freq_sep, N, T):
    t_k = np.real(T * np.log(u_k_est) / freq_sep)
    t_k_wrapped = wrap_tk(t_k, N, T)
    return np.sort(t_k_wrapped)


def nVec(N):
    """ Returns the time interval that we consider [n1*T, (n2+1)*T) """
    if N % 2 == 0:
        n1 = -N / 2
        n2 = N / 2
    else:
        n1 = -(N - 1) / 2
        n2 = (N + 1) / 2

    return np.arange(n1, n2)


def wrap_tk(t_k, N, T):
    """ Wrap t_k back to principal range """

    n_vec = nVec(N)

    t_total = (n_vec[-1]-n_vec[0]+1) * T
    t_start = n_vec[0] * T
    # t_start = 0.5

    t_k_est = np.remainder(t_k-t_start, t_total)+t_start

    return t_k_est