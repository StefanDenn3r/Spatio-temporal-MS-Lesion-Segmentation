import numpy as np
import torch


def asymmetric_loss(beta, output, target):
    g = flatten(target)
    p = flatten(output)
    pg = (p * g).sum(-1)
    beta_sq = beta ** 2
    a = beta_sq / (1 + beta_sq)
    b = 1 / (1 + beta_sq)
    g_p = ((1 - p) * g).sum(-1)
    p_g = (p * (1 - g)).sum(-1)
    loss = (1. + pg) / (1. + pg + a * g_p + b * p_g)
    total_loss = torch.mean(1. - loss)
    return total_loss


def eps_tp_tn_fp_fn(output, target):
    with torch.no_grad():
        epsilon = 1e-7
        target = flatten(target).cpu().detach().float()
        output = flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        tp = torch.sum(target * output)
        tn = torch.sum((1 - target) * (1 - output))
        fp = torch.sum((1 - target) * output)
        fn = torch.sum(target * (1 - output))
        return epsilon, tp.float(), tn.float(), fp.float(), fn.float()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    if type(tensor) == torch.Tensor:
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1).float()
    else:
        return torch.as_tensor(tensor.flatten()).float()
