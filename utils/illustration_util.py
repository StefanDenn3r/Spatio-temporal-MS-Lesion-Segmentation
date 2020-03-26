import cv2
import numpy as np
import torch
from torchvision.utils import make_grid


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res


def visualize_flow(flow):
    """Visualize optical flow

    Args:
        flow: optical flow map with shape of (H, W, 2), with (y, x) order

    Returns:
        RGB image of shape (H, W, 3)
    """
    assert flow.ndim == 3
    assert flow.shape[2] == 2

    hsv = np.zeros([flow.shape[0], flow.shape[1], 3], dtype=np.uint8)
    mag, ang = cv2.cartToPolar(flow[..., 1], flow[..., 0])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def log_visualizations(writer, data, output, target):
    for (a, b, c) in zip(cast(data), cast(output, True), cast(target, True)):
        tensor = np.expand_dims(np.transpose(np.hstack([a, b, c]), (2, 0, 1)), axis=0)
        writer.add_image('input_output_target', make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))


def log_visualizations_longitudinal(writer, x_ref, x, output, target):
    for (a1, a2, b, c) in zip(cast(x_ref), cast(x), cast(output, True), cast(target, True)):
        tensor = np.expand_dims(np.transpose(np.hstack([a1, a2, b, c]), (2, 0, 1)), axis=0)
        writer.add_image('x_ref_x_output_target', make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))


def log_visualizations_deformations(writer, input_moving, input_fixed, flow, target_moving, target_fixed, output=None):
    zipped_data = zip(
        cast(input_moving),
        cast(input_fixed),
        cast(flow, normalize_data=False),
        cast(target_moving, True),
        cast(target_fixed, True),
        cast(output, True) if type(None) != type(output) else [None for _ in input_moving]
    )
    for (_input_moving, _input_fixed, _flow, _target_moving, _target_fixed, _output) in zipped_data:
        transposed_flow = np.transpose(_flow, (1, 2, 0))

        illustration = [
            _input_moving,
            _input_fixed,
            visualize_flow(transposed_flow) / 255.,
            _target_moving,
            _target_fixed
        ]
        if type(None) != type(_output):
            illustration.append(_output)

        tensor = np.expand_dims(np.transpose(np.hstack(illustration), (2, 0, 1)), axis=0)
        description = 'inputmoving_inputfixed_flowfield_targetmoving_targetfixed_output'
        writer.add_image(description, make_grid(torch.as_tensor(tensor), nrow=8, normalize=True))


def cast(data, argmax=False, normalize_data=True):
    data = data.cpu().detach().numpy()
    if argmax:
        data = np.argmax(data, axis=1)

    data = data.astype('float32')

    if normalize_data:
        data = np.asarray([normalize(date) for date in data])

    return data


def normalize(x):
    if len(x.shape) > 2:
        x = x[0]

    return cv2.cvtColor(cv2.normalize(x, None, 0, 1, cv2.NORM_MINMAX), cv2.COLOR_GRAY2RGB)
