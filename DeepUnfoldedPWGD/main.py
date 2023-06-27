import os
import visdom
import numpy as np
import torch
import torch.optim as optim
import argparse
import importlib
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from os.path import join as pjoin

import utils
import utils_data
import utils_model
import loss

# ------------------------------------------------ Parameters ---------------------------------------------------- #
parser = argparse.ArgumentParser(description='Deep Unfolded Projected Wirtinger Gradient Descent (Deep Unfolded PWGD)')
# basic parameters
parser.add_argument('--output_dir', type=str, default='./Model/experiment_name/', help='output directory')
parser.add_argument('--data_dir', type=str, default='./dataset/dataset_dir/', help='data directory')
parser.add_argument('--train_filename', type=str, default="train.h5", help='train data file')
parser.add_argument('--test_filename', type=str, default="test.h5", help='test data file')
parser.add_argument('--c_mn_filename', type=str, help='c_mn_file contain c_mn to be multiplied to y[n] to convert from time domain samples to frequency samples. Will try load s_m directly from .h5 files if not specified')
# Simulation
parser.add_argument('--lr', type=float, default=2e-4, help='Learning Rate')
parser.add_argument('--lr_th', type=float, default=1e-3, help='Learning Rate')
parser.add_argument('--betas', type=float, default=(0.9, 0.999), help='Betas for optimizer')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--drop_last', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Drop the last (incomplete) batch in training')  # >100
# Network layout
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--testbatch_size', type=int, default=1024, help='testing batch size')
# Training Setup
parser.add_argument("--dtype", type=str, default="float", help="float/double, complex numbers")
parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use cuda?')
parser.add_argument("--savestep", type=int, default=50, help="Sets saving step of model parameters")
parser.add_argument("--step", type=int, default=200, help="Sets the learning rate to decay by 10 times every n epochs")
# Simulation Parameters
parser.add_argument('--seed', type=int, default=1000, help='Seed')
parser.add_argument('--model', type=str, help='Choose the model to train')
parser.add_argument('--num_layers', type=int, default=5, help='Number of layers to unfold')
parser.add_argument("--N", type=int, help="Number of Samples")
parser.add_argument("--P", type=int, help="Order of the Kernel = Number of samples in frequency domain-1")
parser.add_argument("--K", type=int, help="Number of Diracs")
parser.add_argument("--th", type=float, default=0.25, help="Initialise soft threshold for singular value thresholding")
parser.add_argument("--P_rank_method", type=str, help="method of projecting to set of low rank matrices: svt/rankK")
parser.add_argument("--loss_fn", type=str, help="SMSE/smMSE/afLoss")
parser.add_argument("--loss_prms", type=float, nargs='+', help="Parameters for loss") #10.0, 0.005 for annihilating filter loss
parser.add_argument('--svl_sizes', type=int, nargs='+', help='Only applies if P_rank_method is a neural network: List of number of hidden neurons per svt layer')
parser.add_argument('--realweight', type=bool, default=False, action=argparse.BooleanOptionalAction, help='use real weights?')
parser.add_argument('--awgn_epoch', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Add noise at each epoch')
parser.add_argument('--awgn_epoch_dist', type=str, default="psnr", help='psnr/sigma')
parser.add_argument('--awgn_epoch_prms', type=float, nargs='*', help='psnr range for adding noise')

prms_in = vars(parser.parse_args())

TORCH_DTYPES = {
    'float': torch.cfloat,
    'double': torch.cdouble
}

# Firstly check if there's an unfinished training (checkpoint)
checkpoint_path = pjoin(prms_in['output_dir'], "model_last.pth")
if os.path.exists(checkpoint_path):
    model, optimizer, lr_scheduler, startEpoch, train_loss_v, test_loss_v, prms = utils.load(checkpoint_path)

    # Update some parameters from argparse in place of loaded model
    prms['device'] = torch.device("cuda") if prms_in['cuda'] else torch.device("cpu")
    prms['num_epochs'] = prms_in['num_epochs']
    prms['output_dir'] = prms_in['output_dir']
    if prms_in['data_dir']:
        prms['data_dir'] = prms_in['data_dir']
        prms['train_filename'] = prms_in['train_filename']
        prms['test_filename'] = prms_in['test_filename']
        prms['c_mn_filename'] = prms_in['c_mn_filename']
    else:
        print('Using path to data files specified previously')

    model = model.to(prms['device'])

    print("=> loaded checkpoint '{} (iter {})'".format(checkpoint_path, startEpoch))
else:
    prms = prms_in
    prms['device'] = torch.device("cuda") if prms_in['cuda'] else torch.device("cpu")
    prms['J'] = utils.getToeplitzJ(prms['P'])   # Number of columns in toeplitz matrix - 1
    prms['rDim'] = prms['P'] - prms['J'] + 1
    prms['cDim'] = prms['J'] + 1

    if isinstance(prms['dtype'], str):
        prms['dtype'] = TORCH_DTYPES[prms['dtype']]

    model_module = getattr(importlib.import_module("model"), prms['model'])
    model = model_module(prms).to(prms['device'])

    optimizer, lr_scheduler = utils_model.set_optimizer_scheduler(model, prms)

    print("=> no checkpoint found at '{}'".format(checkpoint_path))

    startEpoch = 0
    train_loss_v = []
    test_loss_v = []

print("Sampling settings: Number of samples N={:d}, Order of sampling kernel P={:d}, total number of pulses K={:d}".format(prms['N'], prms['P'], prms['K']))

# Select the loss function
loss = loss.loss_sel(prms['loss_fn'], prms, reduce='mean')

utils.set_seed(prms['seed'])
viz = visdom.Visdom()

#########################################################################################
print('===> Loading datasets')
train_data, train_label = utils_data.read_s_m(prms['data_dir'], prms['train_filename'], prms['c_mn_filename'])
test_data, test_label = utils_data.read_s_m(prms['data_dir'], prms['test_filename'], prms['c_mn_filename'])

if prms['awgn_epoch']:
    _, train_data = utils_data.read_y_n(prms['data_dir'], prms['train_filename'])
    c_mn = torch.from_numpy(utils_data.read_c_mn(prms['data_dir'], prms['c_mn_filename'])).to(prms['dtype']).cuda(prms['device'])

num_train_data = train_data.shape[0]
num_test_data = test_data.shape[0]

if prms['P'] != len(train_label[0])-1:
    raise NotImplementedError('P should be number of samples in frequency domain-1')

if prms['awgn_epoch']:
    train_a_k = utils_data.read_a_k(prms['data_dir'], prms['train_filename'])
    train_a_kmax = np.max(np.abs(train_a_k), axis=1)
else:
    train_a_kmax = torch.empty(num_train_data)

train_data = utils_data.toeplitz_S(train_data, prms['J'])
test_data = utils_data.toeplitz_S(test_data, prms['J'])
if 'smMSE' in prms['loss_fn']:
    train_label = train_label
    test_label = test_label
elif 'SMSE' in prms['loss_fn']:
    train_label = utils_data.toeplitz_S(train_label, prms['J'])
    test_label = utils_data.toeplitz_S(test_label, prms['J'])
elif 'afLoss' in prms['loss_fn']: #Annihilating Filter
    train_label = utils_data.read_h(prms['data_dir'], prms['train_filename'])[..., np.newaxis]
    test_label = utils_data.read_h(prms['data_dir'], prms['test_filename'])[..., np.newaxis]


train_set = TensorDataset(torch.from_numpy(train_data).to(prms['dtype']).cuda(prms['device']),
                          torch.FloatTensor(train_a_kmax).cuda(prms['device']),
                          torch.from_numpy(train_label).to(prms['dtype']).cuda(prms['device']))
test_set = TensorDataset(torch.from_numpy(test_data).to(prms['dtype']).cuda(prms['device']),
                         torch.from_numpy(test_label).to(prms['dtype']).cuda(prms['device']))

training_data_loader = DataLoader(train_set, batch_size=prms['batch_size'], shuffle=True, drop_last=prms['drop_last'])
testing_data_loader = DataLoader(test_set, batch_size=prms['testbatch_size'], shuffle=False)

# Make output directory
viz_name = os.path.split(prms['output_dir'])[-1]

if not os.path.exists(prms['output_dir']):
    os.makedirs(prms['output_dir'])

print('===> Building model')
model.train()

# Print the training parameters into a text file
with open(os.path.join(os.path.join(prms['output_dir'], 'train_param.txt')), "w") as f:
    for k, v in sorted(prms.items()):
        f.write("{}: {}\n".format(k, v))

if not os.path.exists(os.path.join(prms['output_dir'], 'results.txt')):
    with open(os.path.join(prms['output_dir'], 'results.txt'), 'w') as f:
        f.write("{}\n".format(os.path.split(prms['output_dir'])[-1]))

print('Learning Rate: ', lr_scheduler.get_last_lr())

for epoch in range(startEpoch+1, prms['num_epochs']+1):

    print('Epoch: %d/%d' % (epoch, prms['num_epochs']))
    pbar = tqdm(total=len(training_data_loader))

    epoch_Trainloss = 0
    epoch_Testloss = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = batch[0].to(prms['device']), batch[-1].to(prms['device'])

        optimizer.zero_grad()

        if prms['awgn_epoch']:  # Noise has to be added onto the y[n] since it might not be Gaussian on s[m]
            a_kmax_batch = batch[-2].to(prms['device'])
            y_n_noisy, _ = utils_model.awgn_psnr(input, a_kmax_batch, psnr_range=prms['awgn_epoch_prms'], dist=prms['awgn_epoch_dist'])
            s_m_noisy = y_n_noisy @ c_mn.T
            input = utils_model.toeplitz_S_torch(s_m_noisy, prms['J']).detach()

        predict = model(input)
        loss_val = loss(predict, target)
        loss_val.backward()

        optimizer.step()

        epoch_Trainloss += loss_val.item()
        pbar.update(1)

    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(prms['device']), batch[-1].to(prms['device'])

            prediction = model(input)
            loss_val = loss(prediction, target)

            epoch_Testloss += loss_val.item()

    epoch_Trainloss = epoch_Trainloss / len(training_data_loader)
    epoch_Testloss = epoch_Testloss / len(testing_data_loader)
    train_loss_v.append(epoch_Trainloss)
    test_loss_v.append(epoch_Testloss)

    pbar.write("===> Epoch {} Complete: Avg. Training Loss: {:.4e},  Avg. Test Loss: {:.4e}".format(epoch, train_loss_v[-1], test_loss_v[-1]))
    pbar.close()

    lr_scheduler.step()
    if epoch % prms['step'] == 0:
        print('Learning Rate: ', lr_scheduler.get_last_lr())

    # Plot the Results
    if epoch == startEpoch+1:
        lossPlotOpts = dict(xlabel='epoch', ylabel='loss', title='Loss', legend=['Training Loss', 'Testing Loss'])
        lossPlot = viz.line(X=np.column_stack((range(1, epoch+1), range(1, epoch+1))),
            Y=np.column_stack((train_loss_v, test_loss_v)),
            env=viz_name, opts=lossPlotOpts)
    else:
        viz.line(X=np.column_stack((epoch, epoch)),
            Y=np.column_stack((epoch_Trainloss, epoch_Testloss)),
            env=viz_name, win=lossPlot, update='append', opts=lossPlotOpts)

    if epoch % prms['savestep'] == 0 or epoch == prms['num_epochs']:
        utils.save(prms['output_dir'], epoch, model, optimizer, lr_scheduler, train_loss_v, test_loss_v, prms)
        with open(os.path.join(prms['output_dir'], 'results.txt'), 'a') as f:
            f.write("===> Epoch {} Complete: Avg. Training Loss: {:.4e},  Avg. Test Loss: {:.4e}\n".format(epoch, epoch_Trainloss, epoch_Testloss))
