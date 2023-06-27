import os
from os.path import join as pjoin
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import argparse

import utils
import utils_data
import utils_model
import prony

# Set printing options
np.set_printoptions(precision=5, suppress=True)

parser = argparse.ArgumentParser("Test the FRIED-Net models, Output a mat file containing the results")

parser.add_argument('--data_dir', type=str, default='./dataset/dataset_dir/', required=True, help='Data directory')
parser.add_argument('--data_filename', type=str, nargs='+', help='test data file')
parser.add_argument('--c_mn_filename', type=str, help='c_mn_file contain c_mn to be multiplied to y[n] to convert from time domain samples to frequency samples. Will try load s_m directly from .h5 files if not specified')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--output_filename', type=str, default=None, help='Output filename, if None then it will be default to be the same as the test data')
parser.add_argument('--model_path', type=str, nargs='+', required=True, help='FRIED-Net file path')
parser.add_argument('--batch_size', type=int, default=100, help='batch size (fine-tuning happens per batch, so typically should be small)')
parser.add_argument('--overwrite', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Overwrite existing results')
parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use cuda/cpu')
# Parameters for Prony's if changing back to time domain
parser.add_argument('--prony', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Run prony\'s method after obtaining the sum of exponentials')
parser.add_argument('--L', type=float, help='2pi*j/L is the separation between reproducible exponential frequencies, e.g. for eMOMS this is P+1')
parser.add_argument("--T", type=float, help="Sampling Period, default is 1/N if unspecified")

args = parser.parse_args()

args.device = torch.device("cuda") if args.cuda else torch.device("cpu")

if len(args.model_path) != len(args.data_filename) and len(args.model_path) != 1:
    raise NotImplementedError('Has to have a single model or the same number of models as number of data files')


for i, file in enumerate(args.data_filename):
    if len(args.model_path) == 1:
        model_path = args.model_path[0]
    else:
        model_path = args.model_path[i]

    model, prms = utils.load(model_path, test=True)
    model = model.to(args.device)
    print('')
    print('Model: ', model_path)

    model.eval()

    if args.output_filename:
        output_file = args.output_filename
    else:
        output_file = file.replace(".h5", ".mat")     # Set output file name to be the same as the data

    output_path = pjoin(args.output_dir, output_file)
    print('Output: ', output_path)

    s_m_noisy, s_m = utils_data.read_s_m(args.data_dir, file, args.c_mn_filename)
    t_k = utils_data.read_t_k(args.data_dir, file)
    num_data = s_m_noisy.shape[0]
    print("Number of Data: ", num_data)

    if not args.overwrite and os.path.exists(output_path):
        mat_contents = sio.loadmat(output_path)
        t_k_est = mat_contents.get('t_k_est', None)
        s_m_est = mat_contents['s_m_est']
        print("Results already exists, loaded from {}".format(output_path))
    else:
        test_data = utils_data.toeplitz_S(s_m_noisy, prms['J'])
        test_set = TensorDataset(torch.from_numpy(test_data).to(prms['dtype']).to(args.device))

        testing_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        pbar = tqdm(total=len(testing_data_loader))

        S_est = torch.Tensor().to(prms['dtype']).to(args.device)
        with torch.no_grad():
            for batch in testing_data_loader:
                input = batch[0].to(args.device)
                tmp = model(input)[-1]
                S_est = torch.cat((S_est, tmp), 0)
                pbar.update(1)
        pbar.close()

        S_est = S_est.detach().cpu()
        s_mR_est, s_mI_est = utils_model.S_to_sm_cp(torch.view_as_real(S_est))
        s_m_est = s_mR_est + 1j * s_mI_est
        s_m_est = s_m_est.numpy()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print("Directory", args.output_dir, "Created")

        if not args.prony:
            sio.savemat(output_path, {'s_m_est': s_m_est})
            t_k_est = None
            print('Saved sum of exponentials')
        else:
            u_k_est, h_est = prony.prony(s_m_est, prms['K'])

            if not args.T:
                args.T = 1 / prms['N']
            t_k_est = prony.freq_to_time(u_k_est, 2j*np.pi/args.L, prms['N'], args.T)

            sio.savemat(output_path, {'s_m_est': s_m_est, 't_k_est': t_k_est, 'N': prms['N'], 'T': args.T, 'K': prms['K']})

        # Print the test parameters into a text file
        with open(pjoin(args.output_dir, 'test_param.txt'), "w") as f:
            for k, v in vars(args).items():
                f.write("{}: {}\n".format(k, v))

    if t_k_est is not None:
        data_idx = np.random.randint(num_data)          # Choose the data to print the results in console
        print('******************** Locations (Data entry {:d}) ********************'.format(data_idx))
        if t_k is not None:
            print('Ground Truth: {}'.format(t_k[0] if len(t_k) == 1 else t_k[data_idx]))
        print('{}: {}'.format(prms['model'], np.squeeze(t_k_est[data_idx])))

        # Calculate the standard deviation for NN and cadzow+prony
        if t_k is not None:
            print('******************** Standard Deviation delta_t ********************')
            try:
                sd_t_k_est = np.mean(np.sqrt((t_k-t_k_est)**2), axis=0)
            except ValueError:
                sd_t_k_est = np.mean(np.sqrt((t_k-np.nan_to_num(t_k_est))**2), axis=0)
            print('{}: {}'.format(prms['model'], sd_t_k_est))
