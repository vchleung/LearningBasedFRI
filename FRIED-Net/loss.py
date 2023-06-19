import torch


def loss_sel(loss_fn, loss_prms=None, reduce='mean'):
    if '+' in loss_fn: # Combine different loss functions separated with +
        return lambda prediction, target: \
            sum(loss_prm * loss_sel(loss)(prediction, target) for loss, loss_prm in zip(loss_fn.split('+'), [1.] + loss_prms))
    else:
        return {
            'ynMSE': lambda prediction, target: MSE(prediction[0], target[0], reduce),
            'ynnoisyMSE': lambda prediction, target: MSE(prediction[0], target[2], reduce),
            'tkMSE': lambda prediction, target: MSE(prediction[1], target[1], reduce),
            'tkChamfer': lambda prediction, target: chamfer_loss(prediction[1].unsqueeze(-1), target[1].unsqueeze(-1), reduce),
            }[loss_fn]


def MSE(pred, target, reduce):
    return torch.nn.functional.mse_loss(pred, target, reduction=reduce)


def batch_pairwise_dist(x: torch.Tensor, y: torch.Tensor):
    """ Adapted from: https://github.com/otaheri/chamfer_distance
    Calculates the pair-wise distance between two sets of points.

    :param x: a tensor of shape ``(m, nx, d)``.
    :param y: a tensor of shape ``(m, ny, d)``.
    :return: the exhaustive distance tensor between every pair of points in `x` and `y`.
    """

    assert x.ndimension() in (2, 3) and y.ndimension() in (2, 3), \
        'Input point clouds must be 2D or 3D tensors'
    if x.ndimension() == 2:
        x = x[None]

    if y.ndimension() == 2:
        y = y[None]

    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))

    diag_ind_x = torch.arange(0, num_points_x).to(device=x.device, dtype=torch.long)
    diag_ind_y = torch.arange(0, num_points_y).to(device=x.device, dtype=torch.long)

    rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P


def chamfer_loss(xyz1, xyz2, reduce='mean'):
    """ Adapted from https://github.com/otaheri/chamfer_distance
    Calculates the Chamfer distance between two batches of point clouds.
    The Pytorch code is adapted from DenseLidarNet_.

    .. _DenseLidarNet: https://github.com/345ishaan/DenseLidarNet/blob/master/code/chamfer_loss.py
    .. _AtlasNet: https://github.com/ThibaultGROUEIX/AtlasNet/tree/master/extension

    :param xyz1: a point cloud of shape ``(b, n1, k)`` or ``(n1, k)``.
    :param xyz2: a point cloud of shape (b, n2, k) or (n2, k).
    :param reduce: ``'mean'`` or ``'sum'``. Default: ``'mean'``.
    :return: the Chamfer distance between the inputs.
    """
    assert len(xyz1.shape) in (2, 3) and len(xyz2.shape) in (2, 3), 'Unknown shape of tensors'

    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)

    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)

    assert reduce in ('mean', 'sum'), 'Unknown reduce method'
    reduce = torch.sum if reduce == 'sum' else torch.mean

    P = batch_pairwise_dist(xyz1, xyz2)
    dist2, _ = torch.min(P, 1)
    dist1, _ = torch.min(P, 2)
    loss_2 = reduce(dist2)
    loss_1 = reduce(dist1)
    return loss_1 + loss_2