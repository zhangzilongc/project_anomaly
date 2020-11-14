#!/usr/bin/python
# -*- coding:utf-8 -*-
from data_loader.base_load import BaseADDataset
from torch.utils.data import DataLoader


class TorchvisionDataset(BaseADDataset):
    """TorchvisionDataset class for datasets already implemented in torchvision.datasets."""

    def __init__(self, root: str):
        super().__init__(root)

    def loaders(self, batch_size: int, shuffle_train=True, shuffle_val=False, num_workers: int = 8, pin_memory=True) -> (
            DataLoader, DataLoader):
        if self.train_status:
            train_loader = DataLoader(dataset=self.train_set, batch_size=batch_size, shuffle=shuffle_train,
                                      num_workers=num_workers, pin_memory=pin_memory)
            val_loader = DataLoader(dataset=self.val_set, batch_size=batch_size, shuffle=shuffle_val,
                                     num_workers=num_workers, pin_memory=pin_memory)
            return train_loader, val_loader
        else:
            test_loader = DataLoader(dataset=self.test_set, batch_size=batch_size, shuffle=shuffle_val,
                                     num_workers=num_workers, pin_memory=pin_memory)
            return test_loader
