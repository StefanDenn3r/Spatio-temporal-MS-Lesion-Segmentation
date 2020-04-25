import torch

from base import BaseModel
from model.FCDenseNet import FCDenseNet
from model.utils.layers import SpatialTransformer


class DeepAtlas(BaseModel):
    def __init__(self, in_channels=8, resolution=(217, 217)):
        super().__init__()
        self.densenet_seg = FCDenseNet(
            in_channels=in_channels, n_classes=1, apply_softmax=False
        )
        self.densenet_voxelmorph = FCDenseNet(
            in_channels=in_channels, n_classes=2, apply_softmax=False
        )
        self.spatial_transform = SpatialTransformer(resolution)

    def forward(self, input_moving, input_fixed):
        x = torch.cat([input_moving, input_fixed], dim=1)
        x_ref = torch.cat([input_fixed, input_moving], dim=1)
        y_seg_moving = torch.sigmoid(self.densenet_seg(x))
        y_seg_fixed = torch.sigmoid(self.densenet_seg(x_ref))
        flow = self.densenet_voxelmorph(x)

        modalities = torch.unbind(input_moving, dim=1)

        y_deformation = torch.stack(
            [
                torch.squeeze(
                    self.spatial_transform(torch.unsqueeze(modality, 1), flow),
                    dim=1
                )
                for modality in modalities
            ],
            dim=1
        )

        y_seg_deformation = self.spatial_transform(y_seg_moving, flow)

        return y_seg_moving, y_seg_fixed, y_deformation, y_seg_deformation, flow
