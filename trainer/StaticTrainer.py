import torch

from trainer.ISBITrainer import ISBITrainer
from utils.illustration_util import log_visualizations


class StaticTrainer(ISBITrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config, data_loader, valid_data_loader, lr_scheduler, len_epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.loss(output, target)
            loss.backward()
            self.optimizer.step()

            self.log_scalars(self.train_metrics, self.get_train_step(batch_idx, epoch), output, target, loss)

            if not (batch_idx % self.log_step):
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(self.data_loader, batch_idx, self.len_epoch)} Loss: {loss.item():.6f}')
                log_visualizations(self.writer, data, output, target)

            del data, target

            if batch_idx == self.len_epoch:
                break
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
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.loss(output, target)

                self.log_scalars(self.valid_metrics, self.get_train_step(batch_idx, epoch), output, target, loss, 'valid')

                if not (batch_idx % self.log_step):
                    self.logger.debug(f'Val Epoch: {epoch} {self._progress(self.valid_data_loader, batch_idx, self.len_epoch_val)} Loss: {loss.item():.6f}')
                    log_visualizations(self.writer, data, output, target)

                del data, target

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
