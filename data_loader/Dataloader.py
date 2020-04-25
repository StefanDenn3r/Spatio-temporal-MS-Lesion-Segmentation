from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate


class Dataloader(DataLoader):
    """
    Data loading
    """

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=1):
        self.dataset = dataset

        self.shuffle = shuffle

        self.batch_idx = 0

        if self.shuffle:
            self.sampler = RandomSampler(self.dataset)
        else:
            self.sampler = SequentialSampler(self.dataset)
        self.shuffle = False

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': default_collate,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)
