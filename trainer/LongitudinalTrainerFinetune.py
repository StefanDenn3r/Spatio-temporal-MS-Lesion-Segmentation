import numpy as np

from trainer.LongitudinalTrainer import LongitudinalTrainer


class LongitudinalTrainerFinetune(LongitudinalTrainer):
    """
    Trainer class
    """

    def __init__(self, model, loss, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        self.start_finetuning = config['trainer']['args']['start_finetuning']
        super().__init__(model, loss, metric_ftns, optimizer, config, data_loader, valid_data_loader, lr_scheduler, len_epoch)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        if self.start_finetuning and (self.not_improved_count == (self.early_stop // 2)):
            self.logger.info(f'Performance has not improved for {self.not_improved_count} Epochs. Unfreezing Encoder...')
            for param in self.model.encoder.parameters():
                param.requires_grad = True

        return super()._train_epoch(epoch)

    def _load_dict(self, checkpoint):
        self.mnt_best = np.inf if self.start_finetuning else checkpoint['monitor_best']
        if self.config['trainer']['args']['only_load_encoder']:
            filtered_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if 'encoder' in k}
            return list(self.model.load_state_dict(filtered_state_dict, False))
        else:
            return list(self.model.load_state_dict(checkpoint['state_dict'], False))
