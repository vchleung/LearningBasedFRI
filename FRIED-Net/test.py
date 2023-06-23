import os
from os.path import join as pjoin
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import argparse

import utils
import utils_data
import utils_model
import loss

# Set printing options
np.set_printoptions(precision=5, suppress=True)

parser = argparse.ArgumentParser("Test the FRIED-Net models, Output a mat file containing the results")

parser.add_argument('--data_dir', type=str, default='./dataset/dataset_dir/', required=True, help='Data directory')
parser.add_argument('--data_filename', type=str, nargs='+', help='test data file')
parser.add_argument('--load_cai_mat', type=bool, default=False, action=argparse.BooleanOptionalAction, help='if the test data is .mat file from calcium imaging data')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
parser.add_argument('--output_filename', type=str, default=None, help='Output filename, if None then it will be default to be the same as the test data')
parser.add_argument('--model_path', type=str, nargs='+', required=True, help='FRIED-Net file path')
parser.add_argument('--batch_size', type=int, default=100, help='batch size (fine-tuning happens per batch, so typically should be small)')
parser.add_argument('--overwrite', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Overwrite existing results')
parser.add_argument('--cuda', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use cuda/cpu')
# Fine-tuning settings
parser.add_argument('--fine_tune', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Fine-tuning the encoder using backpropagation')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--lr', type=float, default=1e-5, help='Learning Rate for locations')
parser.add_argument("--step", type=int, default=150,  help="Sets the learning rate to decay by 10 times every n epochs")

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

    if args.fine_tune:
        model.train()
        utils_model.freeze_layers(model.decoder)      # Only train the encoder
    else:
        model.eval()

    # Set output file name to be the same as the data
    if args.output_filename:
        output_file = args.output_filename
    else:
        output_file = file.replace(".h5", ".mat")
    output_path = pjoin(args.output_dir, output_file)
    print('Output: ', output_path)

    # Import the Test Data
    if args.load_cai_mat:
        y_n_noisy = utils_data.read_y_n_cai(args.data_dir, file, prms['N'])
        t_k = None
    else:
        y_n_noisy, _ = utils_data.read_y_n(args.data_dir, file)
        t_k = utils_data.read_t_k(args.data_dir, file)  # Ground truth locations

    # May only select some range for testing
    # y_n_noisy = y_n_noisy[:1000, :]
    num_data = y_n_noisy.shape[0]
    print("Number of Data: ", num_data)

    if not args.overwrite and os.path.exists(output_path):
        mat_contents = sio.loadmat(output_path)
        t_k_est = mat_contents['t_k_est']
        # a_k_est_NN = mat_contents['inamp_est_batch']
        print("Results already exists, loaded from {}".format(output_path))
    else:
        test_set = TensorDataset(torch.from_numpy(y_n_noisy).float().to(args.device))
        testing_data_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        pbar = tqdm(total=len(testing_data_loader))
        t_k_est = torch.Tensor().to(args.device)

        if args.fine_tune:
            # Save the loaded trained state
            trained_state_dict = model.state_dict().copy()

            for i, batch in enumerate(testing_data_loader):   # Fine-tune for each batch of test data
                y_n_noisy = batch[0].to(args.device)

                optimizer = optim.Adam(model.encoder.parameters(), lr=args.lr)
                decay_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step, gamma=0.1)

                # Reset model_parameters every time
                model.load_state_dict(trained_state_dict)
                if t_k is not None:
                    pbar.write("Example ground truth: {}".format(t_k if t_k.ndim == 1 else t_k[i*args.batch_size]))

                for i in range(args.num_epochs):
                    optimizer.zero_grad()
                    y_recon, t_k_est_batch = model(y_n_noisy, None, None)

                    loss_val = loss.MSE(y_recon, y_n_noisy, reduce='mean')

                    if i % args.step == 0 or i == args.num_epochs - 1:
                        pbar.write(" Avg. Loss: {:.4e}, Location {}".format(loss_val.item(), t_k_est_batch[0].detach().cpu().numpy()))

                    loss_val.backward()
                    optimizer.step()
                    decay_scheduler.step()

                    t_k_est = torch.cat((t_k_est, t_k_est_batch), 0)
                    pbar.update(1)
        else:
            with torch.no_grad():
                for batch in testing_data_loader:
                    y_n_noisy = batch[0].to(args.device)
                    t_k_est_batch = model.encoder(y_n_noisy)
                    t_k_est = torch.cat((t_k_est, t_k_est_batch), 0)
                    pbar.update(1)

        pbar.close()

        if prms['periodic']:
            utils_model.wrap_tk(t_k_est, prms['N'], prms['T'])     # Fix t_k back to principal range if periodic
            t_k_est, _ = torch.sort(t_k_est, dim=-1)         # sort t_k
            print('Periodic signal: Location wrapped and sorted')

        t_k_est = t_k_est.detach().cpu().numpy()

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
            print("Directory", args.output_dir, "Created")

        # Print the test parameters into a text file
        with open(pjoin(args.output_dir, 'test_param.txt'), "w") as f:
            for k, v in vars(args).items():
                f.write("{}: {}\n".format(k, v))

        sio.savemat(output_path, {'t_k_est': t_k_est, 'N': prms['N'], 'T': prms['T'], 'K': prms['K']})

    data_idx = np.random.randint(num_data)          # Choose the data to print the results in console
    print('******************** Locations (Data entry {:d}) ********************'.format(data_idx))
    if t_k is not None:
        print('Ground Truth: {}'.format(t_k[0] if len(t_k) == 1 else t_k[data_idx]))
    print('FRIED-Net {}: {}'.format("(Fine-tuned)" if args.fine_tune else "", np.squeeze(t_k_est[data_idx])))

    # Calculate the standard deviation for NN and cadzow+prony
    if t_k is not None:
        print('******************** Standard Deviation delta_t ********************')
        try:
            sd_t_k_est = np.mean(np.sqrt((t_k-t_k_est)**2), axis=0)
        except ValueError:
            sd_t_k_est = np.mean(np.sqrt((t_k-np.nan_to_num(t_k_est))**2), axis=0)
        print('FRIED-Net {}: {}'.format("(Fine-tuned)" if args.fine_tune else "", sd_t_k_est))
