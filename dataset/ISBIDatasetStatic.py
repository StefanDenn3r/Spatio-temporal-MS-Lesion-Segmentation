import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.dataset_utils import Phase, Modalities, Views, Mode, retrieve_data_dir_paths, Evaluate


class ISBIDatasetStatic(Dataset):
    """ISBIDatasetStatic dataset"""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patient=None, evaluate: Evaluate = Evaluate.TRAINING, preprocess=False,
                 view: Views = None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patient, Mode.STATIC, view)

    def __len__(self):
        return len(self.data_dir_paths)

    def __getitem__(self, idx):
        data, label = [], None
        for i, modality in enumerate(self.modalities):
            with h5py.File(os.path.join(self.data_dir_paths[idx], f'{modality.value}.h5'), 'r') as f:
                data.append(f['data'][()])
                if label is None:
                    label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)

        return torch.as_tensor(np.array(data)), label.float()
