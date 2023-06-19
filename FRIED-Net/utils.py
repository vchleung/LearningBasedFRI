import torch
import numpy as np
import os
import math
import errno
from model import FRIEDNet


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(path_results, epoch, model, optimizer, scheduler, train_loss_v, test_loss_v, prms):

    model_out_path = os.path.join(path_results, "epoch_{:d}.pth".format(epoch))
    cp = os.path.join(path_results, "model_last.pth")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss_v': train_loss_v,
        'test_loss_v': test_loss_v,
        'prms': prms,
    }

    torch.save(checkpoint, model_out_path)
    try:
        symlink_force(model_out_path, cp)
    except OSError:
        torch.save(checkpoint, cp)


def load(file_path, model=None, init="", test=False):
    if os.path.islink(file_path):
        file_path = os.readlink(file_path)

    checkpoint = torch.load(file_path)

    if init:
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if init in k}
        if not pretrained_dict:
            raise NotImplementedError("Nothing loaded in "+init)
        else:
            print(pretrained_dict.keys())
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

    else: # continuing from checkpoint
        prms = checkpoint['prms']

        # Initialise the model, optimizer, scheduler
        model = FRIEDNet(prms, test=test)
        model.load_state_dict(checkpoint['model_state_dict'])

        if test:
            return model, prms
        else:
            optimizer, scheduler = set_optimizer_scheduler(model, prms)
            # Load the checkpoint
            if "optimizer_state_dict" in checkpoint.keys():
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if "scheduler_state_dict" in checkpoint.keys():
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            startEpoch = checkpoint['epoch']
            train_loss_v = checkpoint['train_loss_v']
            test_loss_v = checkpoint['test_loss_v']

            return model, optimizer, scheduler, startEpoch, train_loss_v, test_loss_v, prms


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


def Pack_Matrices_with_NaN(List_of_matrices, Matrix_size=None):
    """
    Extend each vector in array with Nan to reach same shape
    """
    if not Matrix_size:
        Matrix_size = max([max(np.shape(i)) for i in List_of_matrices])
    # print(Matrix_size)
    Matrix_with_nan = np.arange(Matrix_size)
    for array in List_of_matrices:
        array = array.squeeze()
        start_position = np.shape(array)[0]
        for x in range(start_position,Matrix_size):
            array = np.insert(array, (x), np.nan, axis=0)
        Matrix_with_nan = np.vstack([Matrix_with_nan, array])
    Matrix_with_nan = Matrix_with_nan[1:]
    return Matrix_with_nan


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
    """ Returns the time interval that we consider [n1*T, n2*T] """
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