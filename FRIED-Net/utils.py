import torch
import numpy as np
import os
import errno
import utils_model
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