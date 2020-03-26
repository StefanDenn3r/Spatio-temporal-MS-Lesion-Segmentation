import argparse
import os
from pathlib import Path

import nibabel
import numpy as np
import torch
from scipy.ndimage import rotate
from tqdm import tqdm

import data_loader as module_data
import model as module_arch
import model.utils.metric as module_metric
from dataset.ISBIDatasetStatic import Phase
from dataset.dataset_utils import Views
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume=None):
    logger = config.get_logger('test')

    if config.resume:
        resume = config.resume

    resume = Path(resume).parent
    for view in list(Views):
        data_loader = config.retrieve_class('data_loader', module_data)(
            config['data_loader']['args']['data_dir'],
            batch_size=config['data_loader']['args']['batch_size'],
            shuffle=False,
            phase=Phase.TEST,
            modalities=config['data_loader']['args']['modalities'],
            num_workers=config['data_loader']['args']['num_workers'],
            evaluate=config['evaluate'],
            preprocess=config['data_loader']['args']['preprocess'],
            view=view
        )
        resume_path = os.path.join(resume, view.name, 'model_best.pth')
        if os.path.exists(resume_path):
            evaluate_model(config, data_loader, logger, resume_path, view)

    create_final_segmentations(config, logger)
    logger.info('================================')
    logger.info(f'Done')


def create_final_segmentations(config, logger):
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]
    total_metrics = torch.zeros(len(metric_fns))
    patient_metrics = torch.zeros(len(metric_fns))

    save_dir_path = os.path.join(config.config['trainer']['save_dir'], 'output', *str(config._save_dir).split(os.sep)[-2:])
    patient_paths = sorted(os.listdir(save_dir_path))
    for patient_path, patient_dir in zip(map(lambda x: os.path.join(save_dir_path, x), patient_paths), patient_paths):
        if not os.path.isdir(patient_path):
            continue

        segmentation_dirs = sorted(os.listdir(patient_path))
        timestep = 0
        for seg_path, seg_dir in zip(map(lambda x: os.path.join(patient_path, x), segmentation_dirs), segmentation_dirs):
            if not os.path.isdir(seg_path):
                continue
            seg_path_elements = list(map(lambda x: os.path.join(seg_path, x), os.listdir(seg_path)))
            # load all segmentations and average over them
            segmentations = np.round(np.mean(np.asarray(list(map(lambda x: nibabel.load(x).get_data(), seg_path_elements))), axis=0)).astype('int8')
            # save all segmentations with directory's name
            nibabel.save(nibabel.Nifti1Image(segmentations, np.eye(4)), os.path.join(save_dir_path, f'{patient_dir}_{seg_dir}_seg.nii'))
            logger.info(f'Patient {int(patient_path[-2:])} -  Timestep {timestep}:')
            if config['evaluate'] == 'train':
                target_path = os.path.join(config.config['data_loader']['args']['data_dir'], patient_dir, 'masks', f'{patient_dir}_{seg_dir}_mask1.nii')
                target_volume = np.asarray(nibabel.load(target_path).get_data())
                timestep += 1
                for i, metric in enumerate(metric_fns):
                    current_metric = metric(segmentations, target_volume)
                    logger.info(f'      {metric.__name__}: {current_metric}')
                    patient_metrics[i] += current_metric
                    total_metrics[i] += current_metric

        if config['evaluate'] == 'train':
            logger.info(f'Averaged over patient {int(patient_path[-2:])}:')
            for i, met in enumerate(metric_fns):
                logger.info(f'      {met.__name__}: {patient_metrics[i].item() / timestep}')
        patient_metrics = torch.zeros(len(metric_fns))


def evaluate_model(config, data_loader, logger, resume, view):
    # build model architecture
    model = config.initialize_class('arch', module_arch)
    logger.info('Loading checkpoint: {} ...'.format(resume))
    checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        # setup
        patient = 0
        timestep = 0  # max 3
        c = 0
        output_list = []
        n_samples = 0
        timestep_limit = 4
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)
            for slice_output, slice_data, slice_target in zip(output, data, target):
                # we only deal with binary data. Storing only prob for label 1 is enough because of softmax normalization: P(0) = 1 - P(1)
                output_list.append(np.expand_dims(slice_output.cpu().detach().float()[1], axis=0))
                c += 1

                if not c % 217 and c > 0:
                    n_samples += 1
                    path = os.path.join(config.config['trainer']['save_dir'], 'output', *str(config._save_dir).split('/')[-2:])
                    evaluate_timestep(output_list, path, patient, timestep, logger, view, config)

                    # axis = 0
                    timestep += 1
                    if not timestep % timestep_limit and timestep > 0:
                        # inferred whole patient
                        logger.info('---------------------------------')
                        logger.info(f'Done with patient {int(patient) + 1}:')
                        timestep = 0
                        patient += 1
                        if config["evaluate"] == 'test':
                            if patient == 1 or patient == 10 or patient == 13:
                                timestep_limit = 5
                                logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                            elif patient == 9:
                                timestep_limit = 6
                                logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                            else:
                                timestep_limit = 4
                        elif config["evaluate"] == 'training':
                            if patient == 2:
                                timestep_limit = 5
                                logger.info(f'There exist {timestep_limit} timesteps for Patient {int(patient) + 1}')
                            else:
                                timestep_limit = 4

                    output_list = []


def evaluate_timestep(output_list, path, patient, timestep, logger, view, config):
    sub_path = os.path.join(path, f'{config["evaluate"]}{(int(patient) + 1):02}', f'{int(timestep) + 1:02}')
    os.makedirs(sub_path, exist_ok=True)

    seg_volume = np.moveaxis(np.squeeze(np.asarray(output_list)), 0, int(view.value))
    rotated_seg_volume = rotate(rotate(seg_volume, 90, axes=(0, 1)), -90, axes=(1, 2))
    cropped_seg_volume = rotated_seg_volume[18:-18, :, 18:-18]

    nibabel.save(nibabel.Nifti1Image(cropped_seg_volume, np.eye(4)), os.path.join(sub_path, f'{view.name}.nii'))
    logger.info(f'Done with Patient {int(patient) + 1} -  Timestep {int(timestep) + 1:02}')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--evaluate', default='training', type=str, help='Either "training" or "test"; Determines the prefix of the folders to use')

    config = ConfigParser(*parse_cmd_args(args))
    main(config)
