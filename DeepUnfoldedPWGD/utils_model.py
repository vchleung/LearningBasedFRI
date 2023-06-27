import torch
import torch.nn as nn
import math
import utils_data


def set_optimizer_scheduler(model, prms):
    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.named_parameters() if "_th" in name], 'lr': prms['lr_th']},
        {'params': [param for name, param in model.named_parameters() if "_th" not in name], 'lr': prms['lr']}
    ], lr=prms['lr'], betas=prms['betas'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=prms['step'], gamma=0.1)

    return optimizer, lr_scheduler


def init_weights(m, eye=False):
    if eye:
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            nn.init.eye_(m.weight)
    else:
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        elif type(m) == nn.Conv1d:
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='conv1d')


def activation_module_list(hl_sizes, activation, bias=True, dtype=torch.float):
    module_list = []
    module_list.append(nn.Linear(hl_sizes[0], hl_sizes[1], bias=bias).to(dtype))
    for in_f, out_f in zip(hl_sizes[1:], hl_sizes[2:]):
        module_list.append(activation)
        module_list.append(nn.Linear(in_f, out_f).to(dtype))
    return module_list


def awgn_psnr(y, a_kmax, psnr_range, dist):
    """
    batchwise add gaussian noise to signal, PSNR is uniformly distributed between list in psnr_range
    """
    bsz, N = y.shape[0], y.shape[-1]

    if dist == "psnr":
        psnr = psnr_range[0] + (psnr_range[-1]-psnr_range[0]) * torch.rand(bsz, device=y.device, dtype=torch.float)
        tmp = 10 ** (-psnr / 20)
    elif dist == "sigma":
        tmp = torch.rand(bsz, device=y.device, dtype=torch.float)
        psnr = -20 * torch.log10(tmp)

    sigmas = tmp * a_kmax

    noise = torch.randn(y.size(), device=y.device, dtype=y.dtype)
    noiseSqrtPower = torch.linalg.vector_norm(noise, 2, dim=-1) / math.sqrt(N)
    noise = (sigmas/noiseSqrtPower)[:, None] * noise

    return y + noise, psnr


def toeplitz_S_torch(s_m, J):
    """
    FOR BACKPROP
    Args:
        s_m: batched matrix of s_m or s_m
        J: the number of rows - 1
    Returns:
        Batched Toeplitz matrix or a single Toeplitz matrix
    """
    P = s_m.shape[-1] - 1
    m, n = P - J + 1, J + 1

    if s_m.ndim == 1:
        S = torch.empty((m, n), dtype=s_m.dtype, device=s_m.device)
        for p in range(P+1): #iterate for each diagonal
            row_idx, col_idx = kth_diag_indices(m, n, J-p)
            for r, c in zip(row_idx, col_idx):
                S[r, c] = s_m[p]
    elif s_m.ndim == 2:
        num_data = s_m.shape[0]
        S = torch.empty((num_data, m, n), dtype=s_m.dtype, device=s_m.device)
        for p in range(P+1): #iterate for each diagonal
            row_idx, col_idx = kth_diag_indices(m, n, J-p)
            for r, c in zip(row_idx, col_idx):
                S[:, r, c] = s_m[:, p]
    return S


def S_to_sm(S):
    num_data = S.shape[0]
    P = sum(S.shape[1:]) - 2
    J = S.shape[-1]-1

    s_m = torch.empty((num_data, P + 1), device=S.device, dtype=S.dtype)

    for p in range(P + 1):  # iterate for each diagonal
        s_m[:, p] = torch.mean(torch.diagonal(S, J - p, dim1=-2, dim2=-1), dim=-1)

    return s_m


def S_to_sm_cp(S):
    SR, SI = utils_data.split_cp(S)

    s_mR = S_to_sm(SR)
    s_mI = S_to_sm(SI)

    return s_mR, s_mI


def kth_diag_indices(m, n, k):
    """
        Input: number of rows m, number of columns n, diagonal selected k
    """
    if k >= n or k <= -m:
        raise Exception("Diagonals not in range")
    elif k < 0:
        diag_len = min(m+k, n)
        return range(-k, diag_len-k), range(diag_len)
    elif k >= 0:
        diag_len = min(m, n-k)
        return range(diag_len), range(k, diag_len+k)