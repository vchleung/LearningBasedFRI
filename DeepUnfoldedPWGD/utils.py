import torch
import numpy as np
import os
import math
import errno
import importlib
import utils_model


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def getToeplitzJ(P):
    ''' Since toeplitz matrix is better to be near-square, return the number J such that (P-J+1)x(J+1) matrix will be near-square'''
    if P % 2:
        J = int(np.floor((P + 1) / 2 - 1))
    else:
        J = int(np.floor((P + 1) / 2))
    return J


def logit(x):
    y = math.log(x / (1 - x))
    return y


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(output_dir, epoch, model, optimizer, scheduler, train_loss_v, test_loss_v, prms):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_out_path = os.path.join(output_dir, "epoch_{:d}.pth".format(epoch))
    cp = os.path.join(output_dir, "model_last.pth")

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


def load(file_path, model=None, test=False):
    if os.path.islink(file_path):
        file_path = os.readlink(file_path)

    checkpoint = torch.load(file_path)

    # continuing from checkpoint
    prms = checkpoint['prms']

    # Initialise the model, optimizer, scheduler
    model_module = getattr(importlib.import_module("model"), prms['model'])
    model = model_module(prms)
    model.load_state_dict(checkpoint['model_state_dict'])

    if test:
        return model, prms
    else:
        optimizer, scheduler = utils_model.set_optimizer_scheduler(model, prms)
        # Load the checkpoint
        if "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if "scheduler_state_dict" in checkpoint.keys():
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        startEpoch = checkpoint['epoch']
        train_loss_v = checkpoint['train_loss_v']
        test_loss_v = checkpoint['test_loss_v']

        return model, optimizer, scheduler, startEpoch, train_loss_v, test_loss_v, prms



