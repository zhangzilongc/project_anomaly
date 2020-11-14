#!/usr/bin/python
# -*- coding:utf-8 -*-
from models.types import *
from torch import nn
from abc import abstractmethod


class Base_Gray_paint_model(nn.Module):

    def __init__(self) -> None:
        super(Base_Gray_paint_model, self).__init__()

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass
