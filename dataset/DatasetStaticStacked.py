import os

import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from dataset.dataset_utils import Phase, Modalities, Views, retrieve_data_dir_paths, Mode, Evaluate


class DatasetStaticStacked(Dataset):
    """DatasetStaticStacked dataset"""

    def __init__(self, data_dir, phase=Phase.TRAIN, modalities=(), val_patients=None, evaluate: Evaluate = Evaluate.TRAINING, preprocess=True,
                 view: Views = None):
        self.modalities = list(map(lambda x: Modalities(x), modalities))
        self.data_dir_paths = retrieve_data_dir_paths(data_dir, evaluate, phase, preprocess, val_patients, Mode.STATIC, view)

    def __len__(self):
        return len(self.data_dir_paths)

    def __getitem__(self, idx):
        data, label, shape = [], None, 0
        base_path, slice_index = os.path.join(os.sep, *self.data_dir_paths[idx].split(os.sep)[:-1]), int(self.data_dir_paths[idx].split(os.sep)[-1])
        for i, modality in enumerate(self.modalities):
            path = os.path.join(base_path, f'{slice_index - 1:03}', f'{modality.value}.h5')
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    data.append(f['data'][()])
            path = os.path.join(base_path, f'{slice_index:03}', f'{modality.value}.h5')
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    data_ = f['data'][()]
                    shape = data_.shape
                    data.append(data_)
                    if label is None:
                        label = F.one_hot(torch.as_tensor(f['label'][()], dtype=torch.int64), num_classes=2).permute(2, 0, 1)
            path = os.path.join(base_path, f'{slice_index + 1:03}', f'{modality.value}.h5')
            if os.path.exists(path):
                with h5py.File(path, 'r') as f:
                    data.append(f['data'][()])

        if len(data) != len(self.modalities) * 3:
            return torch.zeros(len(self.modalities) * 3, *shape), torch.cat([torch.ones(1, *label.shape[-2:]), torch.zeros(1, *label.shape[-2:])], dim=0)
        return torch.as_tensor(data), label.float()
