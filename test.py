import argparse
import os

import nibabel
import numpy as np
import torch
from scipy.ndimage import rotate
from tqdm import tqdm

import data_loader as module_data_loader
import dataset as module_dataset
import model as module_arch
import model.utils.metric as module_metric
from dataset.ISBIDatasetStatic import Phase
from dataset.dataset_utils import Evaluate
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume=None):
    if config.resume:
        resume = config.resume

    logger = config.get_logger('test')

    # setup data_loader instances
    dataset = config.retrieve_class('dataset', module_dataset)(
        **config['dataset']['args'], phase=Phase.TEST, evaluate=config['evaluate']
    )
    data_loader = config.retrieve_class('data_loader', module_data_loader)(
        dataset=dataset,
        batch_size=config['data_loader']['args']['batch_size'],
        num_workers=config['data_loader']['args']['num_workers'],
        shuffle=False
    )

    # build model architecture
    model = config.initialize_class('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    timestep_limit = 4

    total_metrics = torch.zeros(len(metric_fns))
    patient_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        # setup
        patient = 0
        inner_timestep = 0
        timestep = 0  # max 3
        axis = 0  # max 2
        c = 0

        output_agg = torch.zeros([3, 217, 217, 217]).to(device)
        avg_seg_volume = None
        target_agg = torch.zeros([217, 217, 217]).to(device)

        n_samples = 0
        for idx, loaded_data in enumerate(tqdm(data_loader)):
            if len(loaded_data) == 2:
                inner_timestep_limit = 1
                # static case
                data, target = loaded_data
                data, target = data.to(device), target.to(device)
                output = model(data)
            else:
                # longitudinal case
                inner_timestep_limit = timestep_limit - 1
                x_ref, x, _, target = loaded_data
                x_ref, x, target = x_ref.to(device), x.to(device), target.to(device)
                output = model(x_ref, x)
                if isinstance(output, tuple):
                    output, warp, flow = output

            for slice_output, slice_target in zip(output, target):
                # we only deal with binary data. Storing only prob for label 1 is enough because of softmax normalization: P(0) = 1 - P(1)
                output_agg[axis][c % 217] = torch.unsqueeze(slice_output.float()[1], dim=0)

                if axis == 0 and inner_timestep == 0:
                    target_agg[c % 217] = torch.argmax(slice_target.float(), dim=0)

                c += 1

                if not c % 217 and c > 0:
                    axis += 1
                    if not axis % 3 and axis > 0:
                        path = os.path.join(config.config['trainer']['save_dir'], 'output', *str(config._save_dir).split(os.sep)[-2:],
                                            str(resume).split(os.sep)[-1][:-4])
                        os.makedirs(path, exist_ok=True)

                        if avg_seg_volume is None:
                            avg_seg_volume = get_avg_seg_volume(output_agg)
                        else:
                            avg_seg_volume = torch.cat([avg_seg_volume, get_avg_seg_volume(output_agg)], dim=0)

                        axis = 0
                        inner_timestep += 1
                        if not inner_timestep % inner_timestep_limit and inner_timestep > 0:
                            # inferred one timestep
                            n_samples += 1
                            avg_seg_volume = avg_seg_volume.mean(dim=0)

                            evaluate_timestep(avg_seg_volume, target_agg, metric_fns, config, path, patient, patient_metrics, total_metrics,
                                              timestep,
                                              logger)
                            timestep += 1
                            if not timestep % timestep_limit and timestep > 0:
                                # inferred whole patient
                                logger.info('---------------------------------')
                                logger.info(f'Averaged over patient {int(patient) + 1}:')
                                for i, met in enumerate(metric_fns):
                                    logger.info(f'      {met.__name__}: {patient_metrics[i].item() / timestep}')
                                patient_metrics = torch.zeros(len(metric_fns))
                                timestep = 0
                                patient += 1
                                if config["evaluate"] == Evaluate.TEST:
                                    if patient == 1 or patient == 10 or patient == 13:
                                        timestep_limit = 5
                                        logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                                    elif patient == 9:
                                        timestep_limit = 6
                                        logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                                    else:
                                        timestep_limit = 4
                                elif config["evaluate"] == Evaluate.TRAINING:
                                    if patient == 2:
                                        timestep_limit = 5
                                        logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                                    else:
                                        timestep_limit = 4

                            inner_timestep = 0
                            avg_seg_volume = None

    logger.info('================================')
    logger.info(f'Averaged over all patients:')
    for i, met in enumerate(metric_fns):
        logger.info(f'      {met.__name__}: {total_metrics[i].item() / n_samples}')


def evaluate_timestep(avg_seg_volume, target_agg, metric_fns, config, path, patient, patient_metrics, total_metrics, timestep, logger):
    prefix = f'{config["evaluate"]}{(int(patient) + 1):02}_{int(timestep) + 1:02}'
    seg_volume = torch.round(avg_seg_volume).int().cpu().detach().numpy()
    rotated_seg_volume = rotate(rotate(seg_volume, 90, axes=(0, 1)), -90, axes=(1, 2))
    cropped_seg_volume = rotated_seg_volume[18:-18, :, 18:-18]
    nibabel.save(nibabel.Nifti1Image(cropped_seg_volume, np.eye(4)), os.path.join(path, f'{prefix}_seg.nii'))

    target_volume = torch.squeeze(target_agg).int().cpu().detach().numpy()
    rotated_target_volume = rotate(rotate(target_volume, 90, axes=(0, 1)), -90, axes=(1, 2))
    cropped_target_volume = rotated_target_volume[18:-18, :, 18:-18]
    nibabel.save(nibabel.Nifti1Image(cropped_target_volume, np.eye(4)), os.path.join(path, f'{prefix}_target.nii.gz'))
    # computing loss, metrics on test set
    logger.info(f'Patient {int(patient) + 1} -  Timestep {int(timestep) + 1:02}:')
    for i, metric in enumerate(metric_fns):
        current_metric = metric(cropped_seg_volume, cropped_target_volume)
        logger.info(f'      {metric.__name__}: {current_metric}')
        patient_metrics[i] += current_metric
        total_metrics[i] += current_metric


def get_avg_seg_volume(output_dict):
    axis_volumes = torch.zeros([3, 217, 217, 217])
    for i in range(len(output_dict)):
        axis_volume = torch.squeeze(output_dict[i])
        if i == 1:
            rotated_axis_volume = axis_volume.permute(1, 0, 2)
        elif i == 2:
            rotated_axis_volume = axis_volume.permute(1, 2, 0)
        else:
            rotated_axis_volume = axis_volume
        axis_volumes[i] = rotated_axis_volume

    # Some explanations for the following line:
    # for axis_volumes we only used the predictions for the 1 label. By building the mean over all values up and rounding this we get the value 1
    # for those where the label 1 has the majority in softmax space, else 0. This exactly corresponds to our prediction as we would have taken argmax.
    return torch.unsqueeze(axis_volumes.mean(dim=0), dim=0)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--evaluate', default=Evaluate.TEST, type=Evaluate, help='Either "training" or "test"; Determines the prefix of the folders to use')
    config = ConfigParser(*parse_cmd_args(args))
    main(config)
