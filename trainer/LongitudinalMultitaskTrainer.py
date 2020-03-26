import torch

from trainer.ISBITrainer import ISBITrainer
from utils.illustration_util import log_visualizations_deformations


class LongitudinalMultitaskTrainer(ISBITrainer):
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
        for batch_idx, (input_moving, input_fixed, target_moving, target_fixed) in enumerate(self.data_loader):
            input_moving, input_fixed, target_moving, target_fixed = \
                input_moving.to(self.device), input_fixed.to(self.device), target_moving.to(self.device), target_fixed.to(self.device)

            self.optimizer.zero_grad()
            output, warp, flow = self.model(input_moving, input_fixed)

            # Calculate loss
            loss = self.loss(warp, flow, output, input_fixed, target_fixed)

            loss.backward()
            self.optimizer.step()

            self.log_scalars(self.train_metrics, self.get_train_step(batch_idx, epoch), output, target_fixed, loss)

            if not (batch_idx % self.log_step):
                self.logger.debug(f'Train Epoch: {epoch} {self._progress(self.data_loader, batch_idx, self.len_epoch)} Loss: {loss.item():.6f}')
                log_visualizations_deformations(self.writer, input_moving, input_fixed, flow, target_moving, target_fixed, output)

            del input_moving, input_fixed, target_moving, target_fixed

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
            for batch_idx, (input_moving, input_fixed, target_moving, target_fixed) in enumerate(self.valid_data_loader):
                input_moving, input_fixed, target_moving, target_fixed = \
                    input_moving.to(self.device), input_fixed.to(self.device), target_moving.to(self.device), target_fixed.to(self.device)

                output, warp, flow = self.model(input_moving, input_fixed)

                # Calculate loss
                loss = self.loss(warp, flow, output, input_fixed, target_fixed)

                self.log_scalars(self.valid_metrics, self.get_train_step(batch_idx, epoch), output, target_fixed, loss, 'valid')

                if not (batch_idx % self.log_step):
                    self.logger.debug(f'Val Epoch: {epoch} {self._progress(self.data_loader, batch_idx, self.len_epoch)} Loss: {loss.item():.6f}')
                    log_visualizations_deformations(self.writer, input_moving, input_fixed, flow, target_moving, target_fixed, output)

                del input_moving, input_fixed, target_moving, target_fixed

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()
