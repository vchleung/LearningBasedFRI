import torch
import torch.nn as nn
import utils_model


def svt(x, th, idx=None):
    """
        Soft thresholding singular values
        Input: matrix of size (batch_size, m, n), Output: singular value thresholded estimated matrix of size (batch_size, m, n)
        if idx is specified the threshold is a relative threshold w.r.t. the idx-th singular value, otherwise it's an absolute threshold
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)

    if idx:
        S = torch.relu(S - th * S[:, idx].unsqueeze(-1))
    else:
        S = torch.relu(S - torch.abs(th))

    y = U @ torch.diag_embed(S).to(x.dtype) @ Vh

    return y


def rankK(x, K):
    """
        Thresholding singular values such that the matrix is rank K
        Input: matrix of size (batch_size, m, n), Output: rank-K estimated matrix of size (batch_size, m, n)
    """
    U, S, Vh = torch.linalg.svd(x, full_matrices=False)

    # Only keep the K largest singular values
    Stmp = torch.zeros_like(S)
    Stmp[:, :K] = S[:, :K]

    y = U @ torch.diag_embed(Stmp).to(x.dtype) @ Vh

    return y


class svNet(nn.Module):
    """
        Learn the singular values
    """
    def __init__(self, prms):
        super(svNet, self).__init__()

        num_sv = min(prms['cDim'], prms['rDim'])
        self.device = prms['device']
        self.dtype = prms['dtype']

        self.relu = nn.ReLU(True)

        svl_sizes = [num_sv] + prms['svl_sizes'] + [num_sv]

        tmp = utils_model.activation_module_list(svl_sizes, self.relu)
        tmp.append(self.relu)
        self.svlayers = nn.Sequential(*tmp)

        # Initialise the network
        self.svlayers.apply(lambda x: utils_model.init_weights(x, eye=True))

    def forward(self, x, K, th):

        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        S_out = self.svlayers(S)

        y = U @ torch.diag_embed(S_out).to(x.dtype) @ Vh

        return y


class VNet(nn.Module):
    """
        Learn a vector to multiply with singular values from the singular vector matrix V
    """
    def __init__(self, prms):
        super(VNet, self).__init__()

        self.num_sv = prms['cDim']
        self.device = prms['device']
        self.dtype = prms['dtype']

        self.relu = modReLU(prms)

        svl_sizes = [(self.num_sv**2)] + prms['svl_sizes'] + [self.num_sv]

        tmp = utils_model.activation_module_list(svl_sizes, self.relu, bias=False, dtype=self.dtype)
        self.svlayers = nn.Sequential(*tmp)

        # Initialise the network
        self.svlayers.apply(utils_model.init_weights)

    def forward(self, x, K, th):
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)

        Vhtmp = Vh.detach()
        Vhtmp = Vhtmp.view(Vhtmp.shape[0], -1)

        confidence = torch.sigmoid(self.svlayers(Vhtmp).abs())
        S_out = confidence * S
        y = U @ torch.diag_embed(S_out).to(x.dtype) @ Vh

        return y


class modReLU(nn.Module):
    '''Implementation of modReLU introduced by Arjovsky et al. in Unitary Evolution Recurrent Neural Networks, with learnable threshold'''
    def __init__(self, prms):
        super(modReLU, self).__init__()

        self.device = prms['device']
        self.dtype = prms['dtype']

        self.relu = nn.ReLU()
        self.relubias = nn.Parameter(torch.zeros((1), device=self.device, dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return self.relu(x.abs()+self.relubias) * x / x.abs()

