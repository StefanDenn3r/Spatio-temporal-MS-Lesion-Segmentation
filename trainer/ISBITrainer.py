from abc import ABC

import numpy as np

from base import BaseTrainer
from utils import inf_loop, MetricTracker


class ISBITrainer(BaseTrainer, ABC):
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
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

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

    def get_val_step(self, batch_idx, epoch):
        return (epoch - 1) * len(self.valid_data_loader) + batch_idx

    def get_train_step(self, batch_idx, epoch):
        return (epoch - 1) * self.len_epoch + batch_idx

    def log_scalars(self, metrics, step, output, target, loss, mode='train'):
        self.writer.set_step(step, mode)
        metrics.update('loss', loss.item())
        for met in self.metric_ftns:
            metrics.update(met.__name__, met(output, target))
