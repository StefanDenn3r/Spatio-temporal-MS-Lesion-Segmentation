import os

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.dataset_utils import Phase, Modalities, Mode, retrieve_data_dir_paths, Evaluate


class DatasetLongitudinal(Dataset):
    """DatasetLongitudinal dataset"""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patients=None, evaluate: Evaluate = Evaluate.TRAINING, preprocess=True, view=None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patients, Mode.LONGITUDINAL, view)

    def __len__(self):
        return len(self.data_dir_paths)

    def __getitem__(self, idx):
        x_ref, x, ref_label, label = [], [], None, None
        x_ref_path, x_path = self.data_dir_paths[idx]
        for i, modality in enumerate(self.modalities):
            with h5py.File(os.path.join(x_ref_path, f'{modality.value}.h5'), 'r') as f:
                x_ref.append(f['data'][()])
                if ref_label is None:
                    ref_label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)

            with h5py.File(os.path.join(x_path, f'{modality.value}.h5'), 'r') as f:
                x.append(f['data'][()])
                if label is None:
                    label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)
        return torch.as_tensor(x_ref).float(), torch.as_tensor(x).float(), ref_label.float(), label.float()
