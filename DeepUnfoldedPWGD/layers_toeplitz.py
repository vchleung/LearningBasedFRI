import torch
import utils_model


def averagingDiagonals(x):
    """
        Averaging the diagonals and putting back into Toeplitz matrix
        Input: 2-D matrix of size (m, n) or 3-D matrix of size (batch_size, m, n), Output: estimated Toeplitz matrix
    """
    if x.ndim == 2:
        m, n = x.shape
    elif x.ndim == 3:
        num_data, m, n = x.shape
    else:
        raise NotImplementedError('Tensor has to be 2-D or 3-D')

    P = m+n-2
    S = torch.zeros_like(x)
    for p in range(P+1): #iterate for each diagonal
        row_idx, col_idx = utils_model.kth_diag_indices(m, n, -m+p+1)
        avg = torch.mean(torch.diagonal(x, -m+p+1, dim1=-2, dim2=-1), dim=-1)
        for r, c in zip(row_idx, col_idx):
            S[..., r, c] = avg
    return S