import torch.nn as nn

from base import BaseModel
from model.utils.layers import DenseBlock, TransitionDown, Bottleneck, \
    TransitionUp


class FCDenseNetEncoder(BaseModel):
    def __init__(
            self, in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            bottleneck_layers=4, growth_rate=12, out_chans_first_conv=48
    ):
        super().__init__()
        self.down_blocks = down_blocks
        self.skip_connection_channel_counts = []

        self.add_module(
            'firstconv',
            nn.Conv2d(
                in_channels=in_channels, out_channels=out_chans_first_conv,
                kernel_size=3, stride=1, padding=1, bias=True
            )
        )
        self.cur_channels_count = out_chans_first_conv

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(
                DenseBlock(self.cur_channels_count, growth_rate, down_blocks[i])
            )
            self.cur_channels_count += (growth_rate * down_blocks[i])
            self.skip_connection_channel_counts.insert(
                0, self.cur_channels_count
            )
            self.transDownBlocks.append(TransitionDown(self.cur_channels_count))

        self.add_module(
            'bottleneck',
            Bottleneck(self.cur_channels_count, growth_rate, bottleneck_layers)
        )
        self.prev_block_channels = growth_rate * bottleneck_layers
        self.cur_channels_count += self.prev_block_channels

    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out = self.bottleneck(out)
        return out, skip_connections


class FCDenseNetDecoder(BaseModel):
    def __init__(
            self, prev_block_channels, skip_connection_channel_counts,
            growth_rate, n_classes, up_blocks, apply_softmax=True
    ):
        super().__init__()
        self.apply_softmax = apply_softmax
        self.up_blocks = up_blocks
        self.transUpBlocks = nn.ModuleList([])
        self.denseBlocksUp = nn.ModuleList([])
        for i in range(len(self.up_blocks) - 1):
            self.transUpBlocks.append(
                TransitionUp(prev_block_channels, prev_block_channels)
            )
            cur_channels_count = prev_block_channels + \
                                 skip_connection_channel_counts[i]

            self.denseBlocksUp.append(
                DenseBlock(
                    cur_channels_count, growth_rate, self.up_blocks[i],
                    upsample=True
                )
            )
            prev_block_channels = growth_rate * self.up_blocks[i]
            cur_channels_count += prev_block_channels

        self.transUpBlocks.append(
            TransitionUp(prev_block_channels, prev_block_channels)
        )
        cur_channels_count = prev_block_channels + \
                             skip_connection_channel_counts[-1]
        self.denseBlocksUp.append(
            DenseBlock(
                cur_channels_count, growth_rate, self.up_blocks[-1],
                upsample=False
            )
        )
        cur_channels_count += growth_rate * self.up_blocks[-1]

        self.finalConv = nn.Conv2d(
            in_channels=cur_channels_count, out_channels=n_classes,
            kernel_size=1, stride=1, padding=0, bias=True
        )
        self.softmax = nn.Softmax2d()

    def forward(self, out, skip_connections):
        for i in range(len(self.up_blocks)):
            skip = skip_connections[-i - 1]
            out = self.transUpBlocks[i](out, skip)
            out = self.denseBlocksUp[i](out)

        out = self.finalConv(out)
        if self.apply_softmax:
            out = self.softmax(out)

        return out


class FCDenseNet(BaseModel):
    def __init__(
            self, in_channels=1, down_blocks=(4, 4, 4, 4, 4),
            up_blocks=(4, 4, 4, 4, 4), bottleneck_layers=4, growth_rate=12,
            out_chans_first_conv=48, n_classes=2, apply_softmax=True,
            encoder=None
    ):
        super().__init__()
        self.up_blocks = up_blocks
        self.encoder = encoder
        if not encoder:
            self.encoder = FCDenseNetEncoder(
                in_channels=in_channels, down_blocks=down_blocks,
                bottleneck_layers=bottleneck_layers, growth_rate=growth_rate,
                out_chans_first_conv=out_chans_first_conv
            )

        prev_block_channels = self.encoder.prev_block_channels
        skip_connection_channel_counts = self.encoder.skip_connection_channel_counts

        self.decoder = FCDenseNetDecoder(
            prev_block_channels, skip_connection_channel_counts, growth_rate,
            n_classes, up_blocks, apply_softmax
        )

    def forward(self, x, is_encoder_output=False):
        if is_encoder_output:
            out, skip_connections = x
        else:
            out, skip_connections = self.encoder(x)

        out = self.decoder(out, skip_connections)
        return out
