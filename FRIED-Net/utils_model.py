import torch
import math


def set_optimizer_scheduler(model, prms):
    if not prms['train_encoder']:
        prms['lr_encoder'] = 0.

    if not prms['train_decoder']:
        prms['lr_decoder'] = 0.

    optimizer = torch.optim.Adam([
        {'params': [param for name, param in model.named_parameters() if "encoder" in name],  # Encoder
         'initial_lr': prms['lr_encoder'], 'lr': prms['lr_encoder']},
        {'params': [param for name, param in model.named_parameters() if "decoder" in name],  # Decoder
         'initial_lr': prms['lr_decoder'], 'lr': prms['lr_decoder']},
    ], lr=prms['lr_encoder'], betas=prms['betas'])

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=prms['step'], gamma=0.1)

    return optimizer, lr_scheduler


def freeze_layers(layers):
    """
    Freeze the parameters (Making them unlearnable) in the layers
    """
    for param in layers.parameters():
        param.requires_grad = False


def setParamValue(param, value):
    if param.data.shape == value.shape:
        param.data = value
    else:
        raise NameError('Wrong Shape')


def awgn_psnr(y, a_kmax, psnr_range, dist):
    """
    batchwise add gaussian noise to signal, PSNR is uniformly distributed between list in psnr_range
    """
    bsz, N = y.shape[0], y.shape[-1]

    if dist == "psnr":
        # Sample PSNR from U[psnr_range]
        psnr = psnr_range[0] + (psnr_range[-1]-psnr_range[0]) * torch.rand(bsz, device=y.device, dtype=torch.float)
        tmp = 10 ** (-psnr / 20)
    elif dist == "sigma":
        # Sample sigma from [0,signal_power]
        tmp = torch.rand(bsz, device=y.device, dtype=torch.float)
        psnr = -20 * torch.log10(tmp)

    sigmas = tmp * a_kmax

    noise = torch.randn(y.size(), device=y.device, dtype=y.dtype)
    noiseSqrtPower = torch.linalg.vector_norm(noise, 2, dim=-1) / math.sqrt(N)
    noise = (sigmas/noiseSqrtPower)[:, None] * noise

    return y + noise, psnr


def nVec(N):
    """ Returns the time interval that we consider [n1*T, (n2+1)*T) """
    if N % 2 == 0:
        n1 = -N / 2
        n2 = N / 2
    else:
        n1 = -(N - 1) / 2
        n2 = (N + 1) / 2

    return torch.arange(n1, n2).float()


def wrap_tk(t_k, N, T):
    """ Wrap t_k back to principal range """

    n_vec = nVec(N)

    t_total = (n_vec[-1]-n_vec[0]+1) * T
    t_start = n_vec[0] * T
    # t_start = 0.5

    t_k_est = torch.remainder(t_k-t_start, t_total)+t_start

    return t_k_est