import argparse
import os
from collections import defaultdict
from copy import copy

import numpy as np
import torch

import data_loader as module_data_loader
import dataset as module_dataset
import model as module_arch
import model.utils.loss as module_loss
import model.utils.metric as module_metric
import trainer as trainer_module
from dataset.ISBIDatasetStatic import Phase
from dataset.dataset_utils import Views
from parse_config import ConfigParser, parse_cmd_args


def main(config, resume=None):
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if resume:
        config.resume = resume

    logger = config.get_logger('train')

    # get function handles of loss and metrics
    loss = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # setup data_loader instances
    if config['single_view']:
        results = defaultdict(list)
        for view in list(Views):
            _cfg = copy(config)
            logs = train(logger, _cfg, loss, metrics, view=view)
            for k, v in list(logs.items()):
                results[k].append(v)

    else:
        train(logger, config, loss, metrics)


def train(logger, config, loss, metrics, validation_patient=4, view: Views = None):
    dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.TRAIN, validation_patient=validation_patient, view=view)
    data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'], dataset=dataset)

    val_dataset = config.retrieve_class('dataset', module_dataset)(**config['dataset']['args'], phase=Phase.VAL, validation_patient=validation_patient,
                                                                   view=view)
    valid_data_loader = config.retrieve_class('data_loader', module_data_loader)(**config['data_loader']['args'], dataset=val_dataset)

    # build model architecture, then print to console
    model = config.initialize_class('arch', module_arch)
    logger.info(model)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.initialize('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.initialize('lr_scheduler', torch.optim.lr_scheduler, optimizer)
    if view:
        config._save_dir = os.path.join(config._save_dir, str(view.name))
        config._log_dir = os.path.join(config._log_dir, str(view.name))
        os.mkdir(config._save_dir)
        os.mkdir(config._log_dir)
    trainer = config.retrieve_class('trainer', trainer_module)(model, loss, metrics, optimizer, config, data_loader, valid_data_loader, lr_scheduler)
    return trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args.add_argument('-e', '--evaluate', default='training', type=str, help='Either "training" or "test"; Determines the prefix of the folders to use')
    args.add_argument('-s', '--single_view', default=False, type=bool, help='Defines if a single is used per plane orientation')

    config = ConfigParser(*parse_cmd_args(args))
    main(config)
