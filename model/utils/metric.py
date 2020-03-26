import numpy as np
import torch
from sklearn.metrics import f1_score

from model.utils import metric_utils


def precision(output, target):
    with torch.no_grad():
        epsilon, tp, _, fp, _ = metric_utils.eps_tp_tn_fp_fn(output, target)
        return tp / (tp + fp + epsilon)


def recall(output, target):
    with torch.no_grad():
        epsilon, tp, _, _, fn = metric_utils.eps_tp_tn_fp_fn(output, target)
        return tp / (tp + fn + epsilon)


def dice_loss(output, target):
    with torch.no_grad():
        return metric_utils.asymmetric_loss(1, output, target)


def dice_score(output, target):
    with torch.no_grad():
        target = metric_utils.flatten(target).cpu().detach().float()
        output = metric_utils.flatten(output).cpu().detach().float()
        if len(output.shape) == 2:  # is one hot encoded vector
            target = np.argmax(target, axis=0)
            output = np.argmax(output, axis=0)
        return f1_score(target, output)


def asymmetric_loss(output, target):
    with torch.no_grad():
        return metric_utils.asymmetric_loss(2, output, target)
