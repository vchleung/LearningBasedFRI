import torch
import torch.nn as nn
import layers_rank
import layers_toeplitz
import utils


def P_rank(prms):
    ''' Return the corresponding function handle to perform projection to low rank matrix (of rank K) '''

    if prms['P_rank_method'] == "svt":
        return lambda x, _, exp_th: layers_rank.svt(x, torch.sigmoid(exp_th), 0)
    elif prms['P_rank_method'] == "rankK":
        return lambda x, K, _: layers_rank.rankK(x, K)
    elif prms['P_rank_method'] == "svtK":
        return lambda x, K, exp_th: layers_rank.svt(x, torch.sigmoid(exp_th), int(K-1))
    elif prms['P_rank_method'] == "svtK+1":
        return lambda x, K, exp_th: layers_rank.svt(x, torch.sigmoid(exp_th), int(K))
    elif prms['P_rank_method'] == "svtabs":
        return lambda x, K, exp_th: layers_rank.svt(x, torch.sigmoid(exp_th))
    elif prms['P_rank_method'] == "svNet":
        return layers_rank.svNet(prms)
    elif prms['P_rank_method'] == "VNet":
        return layers_rank.VNet(prms)
    else:
        raise NotImplementedError('No low rank projection selected')


def P_toeplitz(prms):
    return layers_toeplitz.averagingDiagonals


class PWGD(nn.Module):
    def __init__(self, prms):
        super(PWGD, self).__init__()

        rDim = prms['rDim']
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.K = prms['K']

        weights_dtype = torch.float if prms['realweight'] else prms['dtype']

        self.w1 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w2 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w3 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w4 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.exp_th = nn.Parameter(torch.empty((1), device=self.device, dtype=torch.float), requires_grad=True)

        # Initialise with the weights as if operating in wirtinger GD
        delta1 = 0.9999
        delta2 = 0.9999
        self.w1.data.copy_(torch.eye(rDim)*(1-delta1))
        self.w2.data.copy_(torch.eye(rDim)*delta1)
        self.w3.data.copy_(torch.eye(rDim)*delta2)
        self.w4.data.copy_(torch.eye(rDim)*(1-delta2))

        nn.init.constant_(self.exp_th, utils.logit(prms['th']))

        self.P_rank = P_rank(prms)
        self.P_toeplitz = P_toeplitz(prms)

    def forward(self, data):
        L = data[0].to(self.device)
        H = data[1].to(self.device)

        Ltmp = self.w1.to(self.dtype) @ L + self.w2.to(self.dtype) @ H
        Lnew = self.P_rank(Ltmp, self.K, self.exp_th)

        Htmp = self.w3.to(self.dtype) @ Lnew + self.w4.to(self.dtype) @ H
        Hnew = self.P_toeplitz(Htmp)

        return (Lnew, Hnew)


class CadzowUp(nn.Module):
    def __init__(self, prms):
        super(CadzowUp, self).__init__()

        rDim = prms['rDim']
        cDim = prms['cDim']
        self.device = prms['device']
        self.dtype = prms['dtype']
        self.K = prms['K']

        weights_dtype = torch.float if prms['realweight'] else prms['dtype']

        self.w1 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w2 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w3 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.w4 = nn.Parameter(torch.empty((rDim, rDim), device=self.device, dtype=weights_dtype), requires_grad=True)
        self.exp_th = nn.Parameter(torch.empty((1), device=self.device, dtype=torch.float), requires_grad=True)
        self.K = nn.Parameter(torch.empty((1), device=self.device, dtype=torch.int), requires_grad=False)

        self.P_rank = P_rank(prms)
        self.P_toeplitz = P_toeplitz(prms)

        # Initialise with the weights as if operating in wirtinger GD
        nn.init.constant_(self.exp_th, utils.logit(prms['th']))

        self.W = self.generateW(rDim, cDim).to(weights_dtype).to(self.device)

        mu = 0.1
        gamma = 0.51 * mu
        self.w1.data.copy_(torch.eye(rDim) * (1 - gamma))
        self.w2.data.copy_(torch.eye(rDim) * gamma)
        self.w3.data.copy_(mu * self.W)
        self.w4.data.copy_(-mu * self.W)

    def generateW(self, m, n):
        """
            Generating the weight for weighted Frobenius norm for Toeplitz matrix
            Input: (m, n), Output: weight matrix (m, n)
        """
        S = torch.zeros(m, n)
        S[0, :] = 1
        S[:, 0] = 1

        W = layers_toeplitz.averagingDiagonals(S)

        return W

    def forward(self, data):
        T = data[0].to(self.device)
        Tp = data[1].to(self.device)
        Sp = data[2].to(self.device)

        Tptmp = self.w1.to(self.dtype) @ Sp + self.w2.to(self.dtype) @ Tp + self.w4.to(self.dtype) * Tp + self.w3.to(self.dtype) * T

        # Tptmp = Sp + self.w1.to(self.dtype) @ (Tp - Sp) - self.w2.to(self.dtype) @ (self.W * (Tp - T))
        Tpnew = self.P_rank(Tptmp, self.K, self.exp_th)

        Spnew = Sp - Tpnew + self.P_toeplitz(2*Tpnew-Sp)

        return (T, Tpnew, Spnew)


