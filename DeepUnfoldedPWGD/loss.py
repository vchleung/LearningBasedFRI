import torch
import utils_model


def loss_sel(loss_fn, prms=None, reduce='mean'):
    ''' Select Loss function, LH_ means consider loss function a sum of both branches, otherwise only consider output of lower branch H '''
    return {
        'SMSE': lambda prediction, target: MSE(torch.view_as_real(prediction[-1]), torch.view_as_real(target), reduce),
        'smMSE': lambda prediction, target: MSE(torch.stack(utils_model.S_to_sm_cp(torch.view_as_real(prediction[-1])), -1), torch.view_as_real(target), reduce),
        'afLoss': lambda prediction, target: annihilatingFilterLoss(prediction[-1], target, prms, reduce),
        'LH_SMSE': lambda prediction, target: sum([MSE(torch.stack(utils_model.S_to_sm_cp(torch.view_as_real(i)), -1), torch.view_as_real(target), reduce) for i in prediction]) / len(prediction),
        'LH_smMSE': lambda prediction, target: sum([MSE(torch.stack(utils_model.S_to_sm_cp(torch.view_as_real(i)), -1), torch.view_as_real(target), reduce) for i in prediction]) / len(prediction),
        'LH_afLoss': lambda prediction, target: sum([annihilatingFilterLoss(i, target, prms, reduce) for i in prediction]) / len(prediction),
    }[loss_fn]


def MSE(pred, target, reduce):
    return torch.nn.functional.mse_loss(pred, target, reduction=reduce)


def annihilatingFilterLoss(S_pred, h_true, prms, reduce):
    K = prms['K']
    alpha, beta = prms['loss_prms'][0], prms['loss_prms'][1] #10.0, 0.005

    s_pred = torch.stack(utils_model.S_to_sm_cp(torch.view_as_real(S_pred)), -1)
    s_pred = torch.view_as_complex(s_pred)
    S_K = utils_model.toeplitz_S_torch(s_pred, K)
    tmp = S_K @ h_true
    S_comp = S_K - tmp @ h_true.adjoint()
    loss_sum = torch.linalg.vector_norm(tmp.squeeze(-1), dim=-1) ** 2 + alpha * torch.exp(-beta * (torch.linalg.matrix_norm(S_comp, dim=(-1, -2)) ** 2))

    if reduce == 'mean':
        loss_sum = torch.mean(loss_sum)
    elif reduce == 'sum':
        loss_sum = torch.sum(loss_sum)

    return loss_sum
