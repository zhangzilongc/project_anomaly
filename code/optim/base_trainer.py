#!/usr/bin/python
# -*- coding:utf-8 -*-
from abc import ABC, abstractmethod
from data_loader.base_load import BaseADDataset
from models.base import Base_Gray_paint_model


class BaseTrainer(ABC):
    """Trainer base class."""

    def __init__(self, optimizer_name: str, lr: float, n_epochs: int, lr_milestones: tuple, batch_size: int,
                 weight_decay: float, device: str, n_jobs_dataloader: int):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset: BaseADDataset, net: Base_Gray_paint_model):
        """
        Implement train method that trains the given network using the train_set of dataset.
        :return: Trained net
        """
        pass

    # @abstractmethod
    # def test(self, dataset: BaseADDataset, net: BaseVAE):
    #     """
    #     Implement test method that evaluates the test_set of dataset on the given network.
    #     """
    #     pass
