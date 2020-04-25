import torch

from base import BaseModel
from model.FCDenseNet import FCDenseNet, FCDenseNetEncoder
from model.utils.layers import SpatialTransformer


class MultitaskNetwork(BaseModel):
    def __init__(
            self, in_channels=8, resolution=(217, 217)
    ):
        super().__init__()
        self.encoder = FCDenseNetEncoder(in_channels=in_channels)
        self.densenet_seg = FCDenseNet(
            in_channels=in_channels, n_classes=2, apply_softmax=True,
            encoder=self.encoder
        )
        self.densenet_voxelmorph = FCDenseNet(
            in_channels=in_channels, n_classes=2, apply_softmax=False,
            encoder=self.encoder
        )
        self.spatial_transform = SpatialTransformer(resolution)

    def forward(self, input_moving, input_fixed):
        x = torch.cat([input_moving, input_fixed], dim=1)
        out, skip_connections = self.encoder(x)
        y_seg = self.densenet_seg(
            [out, skip_connections], is_encoder_output=True
        )

        flow = self.densenet_voxelmorph(
            [out, skip_connections], is_encoder_output=True
        )

        modalities = torch.unbind(input_moving, dim=1)

        y_deformation = torch.stack(
            [torch.squeeze(
                self.spatial_transform(torch.unsqueeze(modality, 1), flow),
                dim=1
            ) for modality in modalities],
            dim=1
        )
        return y_seg, y_deformation, flow
