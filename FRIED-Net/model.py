import torch
import torch.nn as nn
import layers_encoder
import layers_decoder
import utils_model


def batch_lstsq(phi, y_n, t_k, tstart=-0.5, tend=0.5):
    ''' Our own implementation of least squares estimate on the amplitude to accommodate situations when some locations are out of range '''

    a_k_hat = torch.zeros_like(t_k)

    active_tks = torch.bitwise_and(tstart <= t_k, t_k < tend)   # If t_k is within range = 1, if not = 0 (Active locations)
    for i in range(len(a_k_hat)):
        tmp = phi[i, ..., active_tks[i]]
        a_k_hat[i, active_tks[i]] = torch.linalg.lstsq(tmp, y_n[i]).solution     # tmp.pinverse() @ y_n[i]

    return a_k_hat


def estimate_a_k(true_ak, periodic, N, T, test):
    if true_ak and not test: # No ground truth ak in testing
        return lambda a_k_hat, _, __, ___: a_k_hat
    elif periodic:
        return lambda _, y_n_hat, phi_hat, __: \
            torch.linalg.lstsq(phi_hat.squeeze(), y_n_hat.squeeze().unsqueeze(-1)).solution
    else: # Need to remove effect of the tks that is not in range --> do least squares differently
        n_vec = utils_model.nVec(N)
        tstart, tend = n_vec[0]*T, (n_vec[-1]+1)*T
        return lambda _, y_n_hat, phi_hat, t_k_hat: batch_lstsq(phi_hat, y_n_hat, t_k_hat, tstart, tend)  #detach phi, y for faster computation


class FRIEDNet(nn.Module):
    def __init__(self, prms, test=False):
        super(FRIEDNet, self).__init__()

        if prms['model_encoder']:
            encoderNet = getattr(layers_encoder, prms['model_encoder'])
            self.encoder = encoderNet(prms).to(prms['device'])
        else:
            self.encoder = None

        if prms['model_decoder']:
            decoderNet = getattr(layers_decoder, prms['model_decoder'])
            self.decoder = decoderNet(prms).to(prms['device'])
            self.estimate_a_k = estimate_a_k(prms['true_ak'], prms['periodic'], prms['N'], prms['T'], test)
        else:
            self.decoder = None

    def forward(self, y_noisy, a_k_hat, t_k_init):
        if self.encoder:
            t_k_hat = self.encoder(y_noisy)
        else: # Only train decoder
            t_k_hat = t_k_init

        if self.decoder:
            phi_hat = self.decoder(t_k_hat)
            a_k_hat = self.estimate_a_k(a_k_hat, y_noisy, phi_hat, t_k_hat)
            y_recon = torch.matmul(phi_hat, a_k_hat.squeeze().unsqueeze(-1)).squeeze(-1)
        else: # If no decoder, No y[n] returned
            y_recon = None

        return y_recon, t_k_hat

