import torch
import torch.nn as nn
import layers


class UnfoldedPWGD(nn.Module):
    def __init__(self, prms):
        super(UnfoldedPWGD, self).__init__()

        self.num_layers = prms['num_layers']
        module_list = []

        for i in range(self.num_layers):
            module_list.append(layers.PWGD(prms))

        self.network = nn.Sequential(*module_list)

    def forward(self, S):
        L_0 = torch.zeros_like(S)
        H_0 = S
        data = (L_0, H_0)
        out = self.network(data)

        return out


class UnfoldedCadzowUp(nn.Module):
    def __init__(self, prms):
        super(UnfoldedCadzowUp, self).__init__()

        self.num_layers = prms['num_layers']
        module_list = []

        for i in range(self.num_layers):
            module_list.append(layers.CadzowUp(prms))

        self.network = nn.Sequential(*module_list)

    def forward(self, S):
        T = S
        Tp_0 = S
        Sp_0 = S

        data = (T, Tp_0, Sp_0)
        out = self.network(data)

        return out



