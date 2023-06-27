import os
import matplotlib.pyplot as plt
import visdom
import numpy as np
import torch
import argparse
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import DataLoader, TensorDataset
import json

# Our own codes
import utils
import utils_data
import utils_model
import loss
import phis
from model import FRIEDNet

# Training settings
parser = argparse.ArgumentParser(description='Learning-based FRI: FRI Encoder-Decoder Network')
# basic parameters
parser.add_argument('--output_dir', type=str, default='./Model/experiment_name/', help='output directory')
parser.add_argument('--data_dir', type=str, default='./dataset/dataset_dir/', help='data directory')
parser.add_argument('--train_filename', type=str, default="train.h5", help='train data file')
parser.add_argument('--test_filename', type=str, default="test.h5", help='test data file')
parser.add_argument("--encoder_path", type=str, default=None, help="path to encoder initialisation (only model parameters, not other parameters different from checkpoint)")
parser.add_argument("--decoder_path", type=str, default=None, help="path to decoder initialisation (only model parameters, not other parameters different from checkpoint)")
# Simulation
parser.add_argument('--lr_encoder', type=float, default=1e-4, help='Learning Rate for the encoder')
parser.add_argument('--lr_decoder', type=float, default=1e-5, help='Learning Rate for the decoder (kernel coefficients)')
parser.add_argument('--betas', type=float, default=(0.9, 0.999), help='Betas for optimizer')
parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs')
parser.add_argument('--drop_last', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Drop the last (incomplete) batch in training')
# Network layout
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--testbatch_size', type=int, default=64, help='testing batch size')
# Training Setup
parser.add_argument("--dtype", type=str, default="float32", help="float32/float64")
parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use cuda?')
parser.add_argument("--savestep", type=int, default=50, help="Sets saving step of model parameters")
parser.add_argument("--step", type=int, default=200,  help="Sets the learning rate to decay by 10 times every n epochs")
# Simulation Parameters
parser.add_argument('--seed', type=int, default=1000, help='Seed')
parser.add_argument('--model_encoder', type=str, help='Choose the encoder model (conv)')
parser.add_argument('--model_decoder', type=str, help='Choose the decoder model (decoderReLUNet)')
parser.add_argument("--N", type=int, help="Number of Samples")
parser.add_argument("--K", type=int, nargs='+', help="Number of Diracs")
parser.add_argument("--T", type=float, help="Sampling Period, default is 1/N if unspecified")
parser.add_argument("--samp_mode", type=str, default="anticausal", help="causal/symmetric/peak/anticausal")
parser.add_argument("--loss_fn", type=str, help="ynMSE/ynnoisyMSE/tkMSE/tkChamfer, Use + if combining multiple loss functions")
parser.add_argument("--loss_prms", type=float, nargs='+', help="Parameters for loss functions")
parser.add_argument('--awgn_epoch', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Add noise at each epoch')
parser.add_argument('--awgn_epoch_dist', type=str, default="psnr", help='psnr/sigma')
parser.add_argument('--awgn_epoch_prms', type=float, nargs='*', help='psnr range for adding noise')
# Model Settings
parser.add_argument("--train_encoder", type=bool, default=True, action=argparse.BooleanOptionalAction, help="train encoder")
parser.add_argument("--sort_tk", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Sort t_k in ascending order or not")
parser.add_argument("--train_decoder", type=bool, default=False, action=argparse.BooleanOptionalAction, help="train decoder")
parser.add_argument('--resolution', type=int, default=64, help='How many linear segments within the sampling period T')
parser.add_argument('--periodic', type=bool, default=True, action=argparse.BooleanOptionalAction, help='periodic signal?')
parser.add_argument("--true_ak", type=bool, default=False, action=argparse.BooleanOptionalAction, help="given ground truth amplitude or not (Only applies to decoder)")
parser.add_argument("--true_tk", type=bool, default=False, action=argparse.BooleanOptionalAction, help="Ground truth locations (Only applies when we train decoder only)")
parser.add_argument("--init_phi", type=str, nargs='+', default=[None], help="Initialising/Setting the pulse phi(t) (eMOMS/ESpline/load)")
parser.add_argument("--init_phi_prms", type=json.loads, help="LIST of parameters for init_phi (e.g. order of eMOMS/ESpline)")
parser.add_argument("--norm_phi", type=bool, default=False, action=argparse.BooleanOptionalAction, help="normalise learned phi?")

viz = visdom.Visdom()

prms_in = vars(parser.parse_args())
if not prms_in['T']:
    prms_in['T'] = 1/prms_in['N']
if not isinstance(prms_in['init_phi_prms'], list):
    prms_in['init_phi_prms'] = [prms_in['init_phi_prms']]

TORCH_DTYPES = {
    'float32': torch.float,
    'float64': torch.float64
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
    else:
        print('Using path to data files specified previously')

    model = model.to(prms['device'])

    print("=> loaded checkpoint '{} (iter {})'".format(checkpoint_path, startEpoch))
else:
    prms = prms_in
    prms['K_total'] = sum(prms['K'])
    prms['device'] = torch.device("cuda") if prms_in['cuda'] else torch.device("cpu")
    if isinstance(prms['dtype'], str):
        prms['dtype'] = TORCH_DTYPES[prms['dtype']]

    model = FRIEDNet(prms).to(prms['device'])

    optimizer, lr_scheduler = utils_model.set_optimizer_scheduler(model, prms)

    print("=> no checkpoint found at '{}'".format(checkpoint_path))
    # Initialise model from given file path
    if prms['encoder_path']:
        utils.load(prms['encoder_path'], model, init="encoder")
        print("Encoder Initialised")

    if prms['decoder_path']:
        utils.load(prms['decoder_path'], model, init="decoder")
        print("Decoder Initialised")

    startEpoch = 0
    train_loss_v = []
    test_loss_v = []

if not model.encoder and not model.decoder:
    raise NotImplementedError('No model selected...')

print("Sampling settings: Number of samples N={:d}, Sampling period T={:.4g}, total number of pulses K={:d}".format(prms['N'], prms['T'], prms['K_total']))

loss_fn = loss.loss_sel(prms['loss_fn'], prms['loss_prms'], reduce='mean')   # Select the loss function

torch.set_default_dtype(prms['dtype'])
utils.set_seed(prms['seed'])

#########################################################################################
print('===> Loading datasets')

train_filename = prms['train_filename']
test_filename = prms['test_filename']

train_y_n_noisy, train_y_n = utils_data.read_y_n(prms['data_dir'], train_filename)
test_y_n_noisy, test_y_n = utils_data.read_y_n(prms['data_dir'], test_filename)

train_t_k = utils_data.read_t_k(prms['data_dir'], train_filename)
test_t_k = utils_data.read_t_k(prms['data_dir'], test_filename)

num_train_data = train_y_n_noisy.shape[0]
num_test_data = test_y_n_noisy.shape[0]

if train_y_n is None:
    train_y_n = torch.empty(num_train_data)

if test_y_n is None:
    test_y_n = torch.empty(num_test_data)

print("Training Data / Test Data: ", num_train_data, num_test_data)

if prms['true_ak']:
    train_a_k = utils_data.read_a_k(prms['data_dir'], train_filename)
    test_a_k = utils_data.read_a_k(prms['data_dir'], test_filename)
else:
    train_a_k = torch.empty(num_train_data)
    test_a_k = torch.empty(num_test_data)

if prms['awgn_epoch']:
    tmp = utils_data.read_a_k(prms['data_dir'], train_filename)
    train_a_kmax = np.max(np.abs(tmp), axis=1)      # For calculating PSNR per epoch
else:
    train_a_kmax = torch.empty(num_train_data)

# Make output directory
viz_name = os.path.split(prms['output_dir'])[-1]

if not os.path.exists(prms['output_dir']):
    os.makedirs(prms['output_dir'])


train_set = TensorDataset(torch.Tensor(train_y_n_noisy).type(prms['dtype']).cuda(prms['device']),
                          torch.Tensor(train_a_k).type(prms['dtype']).cuda(prms['device']),
                          torch.Tensor(train_a_kmax).type(prms['dtype']).cuda(prms['device']),
                          torch.Tensor(train_y_n).type(prms['dtype']).cuda(prms['device']),
                          torch.Tensor(train_t_k).type(prms['dtype']).cuda(prms['device']))
test_set = TensorDataset(torch.Tensor(test_y_n_noisy).type(prms['dtype']).cuda(prms['device']),
                         torch.Tensor(test_a_k).type(prms['dtype']).cuda(prms['device']),
                         torch.Tensor(test_y_n).type(prms['dtype']).cuda(prms['device']),
                         torch.Tensor(test_t_k).type(prms['dtype']).cuda(prms['device']))

training_data_loader = DataLoader(train_set, batch_size=prms['batch_size'], shuffle=True, drop_last=prms['drop_last'])
testing_data_loader = DataLoader(test_set, batch_size=prms['testbatch_size'], shuffle=False)

print('===> Building model')

model.train()  # Ready for training

# Freeze encoder/decoder depending on settings
if not prms['train_encoder'] and model.encoder:
    utils_model.freeze_layers(model.encoder)
    print('Encoder Frozen...')

if not prms['train_decoder'] and model.decoder:
    utils_model.freeze_layers(model.decoder)
    print('Decoder Frozen...')

# Print the training parameters into a text file
with open(pjoin(prms['output_dir'], 'train_param.txt'), "w") as f:
    for k, v in sorted(prms.items()):
        f.write("{}: {}\n".format(k, v))

if not os.path.exists(pjoin(prms['output_dir'], 'results.txt')):
    with open(pjoin(prms['output_dir'], 'results.txt'), 'w') as f:
        f.write("{}\n".format(os.path.split(prms['output_dir'])[-1]))

print('Learning Rate: ', lr_scheduler.get_last_lr())

for epoch in range(startEpoch + 1, prms['num_epochs'] + 1):
    if prms['model_decoder']:
        if prms['model_decoder'] == 'decoderReLUNet':
            tmp = [phis.plot_phi_ReLU(model.state_dict()['decoder.c_ReLU.{:d}'.format(i)].detach().cpu(),
                               model.state_dict()['decoder.shift.{:d}'.format(i)].detach().cpu(),
                               model.state_dict()['decoder.init.{:d}'.format(i)].detach().cpu(), prms['resolution'], prms['T']) for i in range(len(prms['K']))]
        elif prms['model_decoder'] == 'decoderTriNet':
            tmp = [phis.plot_phi_tri(model.state_dict()['decoder.c_tri.{:d}'.format(i)].detach().cpu(),
                               model.state_dict()['decoder.shift.{:d}'.format(i)].detach().cpu(), prms['resolution'], prms['T']) for i in range(len(prms['K']))]

        t_phi, phi = list(zip(*tmp))
        t_phi = utils.Pack_Matrices_with_NaN(t_phi).T.squeeze()
        phi = utils.Pack_Matrices_with_NaN(phi).T.squeeze()

        kernelPlotOpts = dict(xlabel='t',title='Phi(t/T)', caption='Epoch {}'.format(epoch - 1),
            legend=["Epoch {}, Phi_{:d}(t)".format(epoch - 1, i) for i in range(len(prms['K']))])
        if epoch == startEpoch + 1:
            kernelPlot = viz.line(X=t_phi, Y=phi, env=viz_name, opts=kernelPlotOpts)
        else:
            viz.line(X=t_phi, Y=phi, env=viz_name, opts=kernelPlotOpts, win=kernelPlot, update='replace')

    print('Epoch: %d/%d' % (epoch, prms['num_epochs']))
    pbar = tqdm(total=len(training_data_loader))

    epoch_Trainloss = 0
    epoch_Testloss = 0

    for batch in training_data_loader:
        a_k_hat0 = batch[1].to(prms['device'])
        y_n = batch[-2].to(prms['device'])
        t_k = batch[-1].to(prms['device'])

        if prms['awgn_epoch']:
            a_kmax_batch = batch[2].to(prms['device'])
            y_n_noisy, _ = utils_model.awgn_psnr(y_n, a_kmax_batch, psnr_range=prms['awgn_epoch_prms'], dist=prms['awgn_epoch_dist'])
        else:
            y_n_noisy = batch[0].to(prms['device'])

        optimizer.zero_grad()

        target = (y_n, t_k, y_n_noisy)

        predict = model(y_n_noisy, a_k_hat0, t_k)
        loss_val = loss_fn(predict, target)
        epoch_Trainloss += loss_val.item()

        loss_val.backward()
        optimizer.step()

        pbar.update(1)

    with torch.no_grad():
        for batch in testing_data_loader:
            y_n_noisy = batch[0].to(prms['device'])
            a_k_hat0 = batch[1].to(prms['device'])
            y_n = batch[-2].to(prms['device'])
            t_k = batch[-1].to(prms['device'])

            target = (y_n, t_k, y_n_noisy)

            predict = model(y_n_noisy, a_k_hat0, t_k)
            loss_val = loss_fn(predict, target)

            epoch_Testloss += loss_val.item()

    epoch_Trainloss = epoch_Trainloss / len(training_data_loader)
    epoch_Testloss = epoch_Testloss / len(testing_data_loader)
    train_loss_v.append(epoch_Trainloss)
    test_loss_v.append(epoch_Testloss)

    pbar.write("===> Epoch {} Complete: Avg. Training Loss: {:.4e},  Avg. Test Loss: {:.4e}".format(epoch,
                                                                                                    train_loss_v[-1],
                                                                                                    test_loss_v[-1]))
    pbar.close()

    lr_scheduler.step()
    if epoch % prms['step'] == 0:
        print(lr_scheduler.get_last_lr())

    # Plot the Results
    if epoch == startEpoch + 1:
        lossPlotOpts = dict(xlabel='epoch', ylabel='loss', title='Loss', legend=['Training Loss', 'Testing Loss'])
        lossPlot = viz.line(X=np.column_stack((range(1, epoch + 1), range(1, epoch + 1))),
                            Y=np.column_stack((train_loss_v, test_loss_v)),
                            env=viz_name, opts=lossPlotOpts)

    else:
        viz.line(X=np.column_stack((epoch, epoch)),
                Y=np.column_stack((epoch_Trainloss, epoch_Testloss)),
                env=viz_name, win=lossPlot, update='append', opts=lossPlotOpts)

    if epoch % prms['savestep'] == 0 or epoch == prms['num_epochs']:
        utils.save(prms['output_dir'], epoch, model, optimizer, lr_scheduler, train_loss_v, test_loss_v, prms)
        with open(pjoin(prms['output_dir'], 'results.txt'), 'a') as f:
            f.write("===> Epoch {} Complete: Avg. Training Loss: {:.4e},  Avg. Test Loss: {:.4e}\n".format(epoch, epoch_Trainloss, epoch_Testloss))

        if prms['train_decoder'] and t_phi is not None and phi is not None:
            plt.figure(figsize=(6, 2), dpi=150)
            plt.plot(t_phi, phi)
            plt.tight_layout()
            plt.savefig(pjoin(prms['output_dir'], "learned_phi.png"))
