from logger import Mode
from trainer.Trainer import Trainer
from utils.illustration_util import log_visualizations_deformations


class LongitudinalMultitaskTrainer(Trainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, loss, metric_ftns, optimizer, config, data_loader, valid_data_loader, lr_scheduler, len_epoch)

    def _process(self, epoch, data_loader, metrics, mode: Mode = Mode.TRAIN):
        _len_epoch = self.len_epoch if mode == Mode.TRAIN else self.len_epoch_val
        for batch_idx, (input_moving, input_fixed, target_moving, target_fixed) in enumerate(data_loader):
            input_moving, input_fixed, target_moving, target_fixed = \
                input_moving.to(self.device), input_fixed.to(self.device), target_moving.to(self.device), target_fixed.to(self.device)

            if mode == Mode.TRAIN:
                self.optimizer.zero_grad()

            output, warp, flow = self.model(input_moving, input_fixed)
            loss = self.loss(warp, flow, output, input_fixed, target_fixed)

            if mode == Mode.TRAIN:
                loss.backward()
                self.optimizer.step()

            self.log_scalars(metrics, self.get_step(batch_idx, epoch, _len_epoch), output, target_fixed, loss, mode)

            if not (batch_idx % self.log_step):
                self.logger.info(f'{mode.value} Epoch: {epoch} {self._progress(data_loader, batch_idx, _len_epoch)} Loss: {loss.item():.6f}')
            if not (batch_idx % (_len_epoch // 10)):
                log_visualizations_deformations(self.writer, input_moving, input_fixed, flow, target_moving, target_fixed, output)

            del input_moving, input_fixed, target_moving, target_fixed
