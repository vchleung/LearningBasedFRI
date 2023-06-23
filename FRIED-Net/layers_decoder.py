import torch
import torch.nn as nn
import numpy as np
import copy
import phis
import utils_model


def periodicSamp(periodic_samp, N):
    """ Periodic signal/sampling: return remainder(t_k/T-n,N) """
    if periodic_samp:
        return lambda data: torch.remainder(data, N)
    else:
        return lambda data: data


def computePeakPhi(c, init=0, resolution=64., model_decoder="ReLU"):
    """ Return the peak (positive or negative) value of the pulse phi """

    if model_decoder == "ReLU":
        phi = torch.cumsum(torch.cumsum(c, dim=0), dim=0) / resolution + init
    elif model_decoder == "tri":
        phi = c

    if phi.max().abs() >= phi.min().abs():
        return phi.max()
    else:
        return phi.min()


def phiInit(init_phi, resolution, N, init_prms, samp_mode):
    ''' Choose the initialisation of phi '''

    if not init_phi:
        t_phi = np.arange(0, N, 1 / resolution)
        phi = np.zeros_like(t_phi)
    elif init_phi == "uniform":     #random value
        t_phi = np.arange(0, N, 1 / resolution)
        phi = np.random.uniform(-0.01, 0.01, size=t_phi.shape)
    elif "load" in init_phi:
        t_phi, phi = phis.load_phi(init_prms)
    else:
        phi_fn = getattr(phis, init_phi)
        if isinstance(init_prms, list):
            t_phi, phi = phi_fn(resolution, *init_prms)
        else:
            t_phi, phi = phi_fn(resolution, init_prms)

    if samp_mode == "symmetric":
        t_mid = (t_phi[-1] - t_phi[0]) / 2
        t_phi = t_phi - t_mid
    elif samp_mode == "peak":   #The peak of the pulse = center
        i_max = np.argmax(np.abs(phi))
        t_phi = t_phi - t_phi[i_max]
    elif samp_mode == "causal":
        t_phi = np.flip(-t_phi)
        phi = np.flip(phi)

    return t_phi, phi


def linearApproxReLU(t_phi, phi, resolution):

    phi = np.array(phi).squeeze()
    t_phi = np.array(t_phi).squeeze()

    T_s = t_phi[1]-t_phi[0]
    stride = int(1/(T_s * resolution))  # Signal resolution divided by desired resolution
    T_int = 1/resolution

    b_ReLU = t_phi[::stride]
    b_ReLU = np.append(b_ReLU, b_ReLU[-1]+T_int)
    b_ReLU = -b_ReLU

    phi = np.append(phi[::stride], phi[0])       # periodically extend by 1 to allow calculation of diff
    init = np.array(phi[0])
    delta_phi = np.diff(phi)            # Find the change in the value of phi between two time steps

    grad = delta_phi/T_int              # Find the gradient

    c_ReLU = np.diff(grad)
    c_ReLU = np.append(grad[0], c_ReLU)          # Add the initial conditions
    c_ReLU = np.append(c_ReLU, -sum(c_ReLU))     # Add the last ReLU to let all sum to zero

    knots = c_ReLU.shape[0]

    return c_ReLU, b_ReLU, init, knots


def linearApproxTri(t_phi, phi, resolution):

    phi = np.array(phi).squeeze()
    t_phi = np.array(t_phi).squeeze()

    T_s = t_phi[1]-t_phi[0]
    stride = int(1/(T_s * resolution))  # Signal resolution divided by desired resolution
    T_int = 1/resolution

    b_tri = t_phi[::stride]
    b_tri = np.insert(b_tri, 0, b_tri[0] - T_int)
    b_tri = np.append(b_tri, b_tri[-1] + T_int)
    b_tri = -b_tri

    c_tri = phi[::stride]/T_int

    knots = b_tri.shape[0]

    return c_tri, b_tri, knots


class decoderReLUNet(nn.Module):
    """ Decoder mapping t_k to samples y[n] using ReLU to perform linear interpolation """
    def __init__(self, prms):
        super(decoderReLUNet, self).__init__()

        self.N = prms['N']
        self.K = prms['K']
        self.T = prms['T']
        self.resolution = prms['resolution']
        self.device = prms['device']
        self.train_decoder = prms['train_decoder']
        self.norm_phi = prms['norm_phi']

        self.periodicSamp = periodicSamp(prms['periodic'], self.N)

        self.n_vec = utils_model.nVec(self.N).to(self.device)

        self.c_ReLU = nn.ParameterList()
        self.shift = nn.ParameterList()
        self.init = nn.ParameterList()
        self.knots = nn.ParameterList()
        self.knots_tot = 0

        for i, k in enumerate(self.K):
            t_phi, phi = phiInit(prms['init_phi'][i], self.resolution, self.N, prms['init_phi_prms'][i], prms['samp_mode'])

            tmp = linearApproxReLU(t_phi, phi, self.resolution)
            self.shift.extend([nn.Parameter(torch.from_numpy(tmp[1]).float().to(self.device), requires_grad=False)])
            self.knots.extend([nn.Parameter(torch.FloatTensor([tmp[3]]).to(self.device), requires_grad=False)])
            self.knots_tot += tmp[3] * k
            self.c_ReLU.extend([nn.Parameter(torch.from_numpy(tmp[0]).float().to(self.device), requires_grad=bool(self.train_decoder))])
            self.init.extend([nn.Parameter(torch.from_numpy(tmp[2]).float().to(self.device), requires_grad=False)])

            if not prms['init_phi'][i]:
                # Randomly initialise the coefficients if no phi specified
                torch.nn.init.uniform_(self.c_ReLU[i], a=-0.01, b=0.01)

        self.c_ReLU_plot = copy.deepcopy(self.c_ReLU)
        self.c_ReLU_plot.requires_grad_(False)

        layersList=[nn.Linear(prms['K_total'], self.knots_tot),
            nn.ReLU(True),
            ]

        self.layers = nn.Sequential(*layersList)

        self.init_decoder_weights()
        utils_model.freeze_layers(self.layers)

    def init_decoder_weights(self):
        """ initialise the parameters for neural network """
        x = []
        y = []
        for knot, shift, k in zip(self.knots, self.shift, self.K):
            x.extend([torch.ones(int(knot.data))] * k)
            y.extend([shift.data] * k)

        layer2weight = torch.block_diag(*x).float().t().to(self.device)

        layer2bias = torch.cat(y, 0).float().to(self.device)

        utils_model.setParamValue(self.layers[0].weight, layer2weight)
        utils_model.setParamValue(self.layers[0].bias, layer2bias)

    def forward(self, t_k_hat):

        # Layer 1 (t_k===>t_k/T-n)
        fc1 = self.periodicSamp(t_k_hat.unsqueeze(-1)/self.T - self.n_vec.to(t_k_hat.device)) #(num_data, K, N)
        fc1 = fc1.transpose(-1, -2)    #(num_data, N, K)

        # Layer 2 (t_k/T-n --> ReLU(t_k/T-n-i))
        fc2 = self.layers(fc1)         #(num_data, N, knots_tot)

        # Layer 3 (phi(x) = c_i*ReLU(x-i))
        lst = []
        lst_bias = []
        for i, j, k, pl in zip(self.c_ReLU, self.init, self.K, self.c_ReLU_plot):
            if self.train_decoder:
                # i = i.clone()
                if self.norm_phi:
                    # Normalise the coefficients by dividing them by the peak value of phi
                    tmp = computePeakPhi(i, j, self.resolution)
                    i = i / tmp
                    j = j / tmp
                i.data[-1] = -torch.sum(i[:-1])
            lst.extend([i] * k)
            lst_bias.extend([j] * k)
            pl.data = i

        layer3weight = torch.block_diag(*lst).float().to(t_k_hat.device)
        layer3bias = torch.Tensor(lst_bias).float().to(t_k_hat.device)

        phi_hat = torch.matmul(fc2, layer3weight.t()) + layer3bias  #(num_data, N, K)

        return phi_hat


class decoderTriNet(nn.Module):
    """ Decoder mapping t_k to samples y[n] using 1st order spline (triangular function), making each coefficient independent to one another"""
    def __init__(self, prms):
        super(decoderTriNet, self).__init__()

        print('decoderTriNet')
        self.N = prms['N']
        self.K = prms['K']
        self.T = prms['T']
        self.resolution = prms['resolution']
        self.device = prms['device']
        self.train_decoder = prms['train_decoder']
        self.norm_phi = prms.get('norm_phi', False)
        self.dtype = prms['dtype']

        self.periodicSamp = periodicSamp(prms['periodic'], self.N)

        self.n_vec = utils_model.nVec(self.N).to(self.device)

        self.c_tri = nn.ParameterList()
        self.shift = nn.ParameterList()
        self.knots = []
        self.remove_idx = []
        self.knots_tot = 0
        self.T_int = 1/self.resolution

        for i, k in enumerate(self.K):
            t_phi, phi = phiInit(prms['init_phi'][i], self.resolution, self.N, prms['init_phi_prms'][i], prms['samp_mode'])

            tmp = linearApproxTri(t_phi, phi, self.resolution)
            self.shift.extend([nn.Parameter(torch.from_numpy(tmp[1]).type(self.dtype).to(self.device), requires_grad=False)])
            self.knots.append(tmp[2])
            self.remove_idx.extend([tmp[2]] * k)
            self.knots_tot += tmp[2] * k
            self.c_tri.extend([nn.Parameter(torch.from_numpy(tmp[0]).type(self.dtype).to(self.device), requires_grad=bool(self.train_decoder))])

        self.remove_idx = (np.cumsum(self.remove_idx)-1).tolist() + (np.cumsum(self.remove_idx)-2).tolist()

        self.select_idx = [i for i in range(self.knots_tot) if i not in self.remove_idx]

        if self.train_decoder and not prms['init_phi']:
            # Randomly initialise the coefficients
            for param in self.c_tri:
                torch.nn.init.uniform_(param, a=0, b=1/self.T_int)

        self.c_tri_plot = copy.deepcopy(self.c_tri)
        self.c_tri_plot.requires_grad_(False)

        layersList=[nn.Linear(self.K, self.knots_tot),
            nn.ReLU(True),
            nn.Conv1d(self.N, self.N, 3, groups=self.N, bias=False)
            ]

        self.layers = nn.Sequential(*layersList)

        self.init_decoder_weights()
        utils_model.freeze_layers(self.layers)

    def init_decoder_weights(self):
        x = []
        y = []
        for knot, shift, k in zip(self.knots, self.shift, self.K):
            x.extend([torch.ones(knot)] * k)
            y.extend([shift.data] * k)

        layer2weight = torch.block_diag(*x).t().type(self.dtype).to(self.device)
        layer2bias = torch.cat(y, 0).type(self.dtype).to(self.device)

        utils_model.setParamValue(self.layers[0].weight, layer2weight)
        utils_model.setParamValue(self.layers[0].bias, layer2bias)

        layer3weight = torch.tensor([1., -2., 1.], dtype=self.dtype, requires_grad=False).repeat(self.N, 1, 1).to(self.device) # Triangular function in terms of ReLU

        utils_model.setParamValue(self.layers[2].weight, layer3weight)

    def forward(self, t_k_hat):

        # Layer 1 (t_k===>t_k/T-n)
        fc1 = self.periodicSamp(t_k_hat.unsqueeze(-1)/self.T - self.n_vec) #(num_data, K, N)
        fc1 = fc1.transpose(-1, -2)    #(num_data, N, K)

        # Layer 2 (t_k/T-n --> ReLU(t_k/T-n-i))
        fc2 = self.layers(fc1)         #(num_data, N, knots_tot)

        # Layer 3 (ReLU(x) --> , phi(x) = c_i*triangle(x-i))
        lst = []
        for i, k, pl in zip(self.c_tri, self.K, self.c_tri_plot):
            if self.train_decoder:
                # print(i.data)
                # i = (i + torch.cat((i[1:], torch.zeros(1, device=self.device))))/2
                if self.norm_phi:
                    i = i / (computePeakPhi(i, resolution=self.resolution, model_decoder="tri") * self.T_int)
            lst.extend([i] * k)
            pl.data = i

        layer3weight = torch.block_diag(*lst).to(self.device).type(self.dtype)

        phi_hat = torch.matmul(fc2[..., self.select_idx], layer3weight.t())

        return phi_hat
