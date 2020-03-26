from base import BaseModel
from model.FCDenseNet import FCDenseNetEncoder, FCDenseNetDecoder
from model.utils.layers import *


class LongitudinalFCDenseNet(BaseModel):
    def __init__(self,
                 in_channels=1, down_blocks=(4, 4, 4, 4, 4),
                 up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4,
                 growth_rate=12, out_chans_first_conv=48, n_classes=2, encoder=None, siamese=True):
        super().__init__()
        self.up_blocks = up_blocks
        self.densenet_encoder = encoder
        self.siamese = siamese
        if not encoder:
            self.densenet_encoder = FCDenseNetEncoder(in_channels=in_channels * (1 if siamese else 2), down_blocks=down_blocks,
                                                      bottleneck_layers=bottleneck_layers,
                                                      growth_rate=growth_rate, out_chans_first_conv=out_chans_first_conv)

        prev_block_channels = self.densenet_encoder.prev_block_channels
        skip_connection_channel_counts = self.densenet_encoder.skip_connection_channel_counts

        if self.siamese:
            self.add_module('merge_conv', nn.Conv2d(prev_block_channels * 2, prev_block_channels, 1, 1))

        self.decoder = FCDenseNetDecoder(prev_block_channels, skip_connection_channel_counts, growth_rate, n_classes, up_blocks)

    def forward(self, x_ref, x):
        if self.siamese:
            out, skip_connections = self.densenet_encoder(x)
            out_ref, _ = self.densenet_encoder(x_ref)
            out = torch.cat((out, out_ref), dim=1)
            out = self.merge_conv(out)
        else:
            out, skip_connections = self.densenet_encoder(torch.cat((x_ref, x), dim=1))

        out = self.decoder(out, skip_connections)

        return out
