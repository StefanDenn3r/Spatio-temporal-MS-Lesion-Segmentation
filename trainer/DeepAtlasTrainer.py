import torch

from logger import Mode
from trainer.Trainer import Trainer
from utils.illustration_util import log_visualizations_deformations


class DeepAtlasTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(
            model, loss, metric_ftns, optimizer, config, data_loader,
            valid_data_loader, lr_scheduler, len_epoch
        )

    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        _len_epoch = self.len_epoch if mode == Mode.TRAIN else self.len_epoch_val
        for batch_idx, (x_ref, x, target_ref, target) in enumerate(data_loader):
            x_ref, x, target_ref, target = x_ref.to(self.device), x.to(self.device), target_ref.to(self.device), target.to(self.device)

            if mode == Mode.TRAIN:
                self.optimizer.zero_grad()
            output = self.model(x_ref, x)

            loss, scores = self.loss(
                *output,
                torch.argmax(target_ref, dim=1, keepdim=True).float(),
                torch.argmax(target, dim=1, keepdim=True).float(),
                x
            )
            if mode == Mode.TRAIN:
                loss.backward()
                self.optimizer.step()

            self.log_scalars(
                metrics,
                self.get_step(batch_idx, epoch, _len_epoch),
                output[1],
                torch.argmax(target, dim=1, keepdim=True),
                scores,
                mode
            )

            if not (batch_idx % self.log_step):
                self.logger.info(
                    f'{mode.value} Epoch: {epoch} '
                    f'{self._progress(data_loader, batch_idx, _len_epoch)} '
                    f'Loss: {loss.item():.6f}'
                )
            if not (batch_idx % (_len_epoch // 10)):
                log_visualizations_deformations(
                    self.writer, x_ref, x, output[-1], target_ref, target
                )

            del x_ref, x, target
