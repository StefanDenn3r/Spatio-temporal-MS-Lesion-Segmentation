from abc import abstractmethod

import numpy as np
import torch

from base import BaseTrainer
from logger import Mode
from utils import MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
            self.len_epoch_val = len(self.valid_data_loader) if self.do_validation else 0

        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    @abstractmethod
    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        raise NotImplementedError('Method _process() from Trainer class has to be implemented!')

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()

        self._process(epoch, self.data_loader, self.train_metrics, Mode.TRAIN)

        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            self._process(epoch, self.valid_data_loader, self.valid_metrics, Mode.VAL)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def log_scalars(self, metrics, step, output, target, loss, mode=Mode.TRAIN):
        self.writer.set_step(step, mode)
        metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            metrics.update(met.__name__, met(output, target))

    @staticmethod
    def _progress(data_loader, batch_idx, batches):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = batches
        return base.format(current, total, 100.0 * current / total)

    @staticmethod
    def get_step(batch_idx, epoch, len_epoch):
        return (epoch - 1) * len_epoch + batch_idx
