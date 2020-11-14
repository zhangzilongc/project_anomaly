#!/usr/bin/python
# -*- coding:utf-8 -*-
from data_loader.torchvision_dataset import TorchvisionDataset
from torch.utils.data import Dataset
import cv2

import torchvision.transforms as transforms
from glob import glob
import os
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import numpy as np
import random
import logging
import math


class ZT_Dataset(TorchvisionDataset):

    def __init__(self, root: str, image_size=128, train_interval=128,
                 train_status=True, train_bins_interval=8, part1_train=False, part2_train=False,
                 part3_train=False, part3_test=False, train_test_normal=False, frac=1, noise_level=3, edge_lower=15,
                 edge_up=30, edge_lower_part3=6, edge_up_part3=15, part_figure=42000, data_aug=False):
        super().__init__(root)

        root_list = glob(os.path.join(self.root, '*.bmp'))#[index_order_start: index_order_end]

        if data_aug:
            num_sample = 1.5
        else:
            num_sample = 1

        if train_status:
            train_list, val_list = train_test_split(root_list, test_size=0.2, random_state=6)
            temp_root = train_list[0]
            ori_ima = cv2.imread(temp_root)
            ima_row, ima_col = ori_ima.shape[0], ori_ima.shape[1]
            floor_interval = math.floor(np.sqrt(ima_row * ima_col * len(train_list) / (part_figure / num_sample)))
            if floor_interval > train_interval:
                frac = (part_figure / num_sample) / len(train_list) / (
                            (ima_row / train_interval) * (ima_col / train_interval))
                frac *= 1.35
            else:
                train_interval = floor_interval

        transform = transforms.Compose([transforms.ToPILImage(),
                                        #transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()])
        self.train_status = train_status
        self.train_test_normal = train_test_normal

        if train_status:
            self.train_set = MyZT_dataset(train_list, transform, train_status=train_status,
                                          image_size=image_size, train_interval=train_interval,
                                          train_bins_interval=train_bins_interval, part1_train=part1_train,
                                          part2_train=part2_train, part3_train=part3_train,
                                          train_test_normal=train_test_normal, frac=frac, noise_level=noise_level,
                                          edge_lower=edge_lower, edge_up=edge_up,
                                          edge_lower_part3=edge_lower_part3, edge_up_part3=edge_up_part3,
                                          part_figure=part_figure, data_aug=data_aug)

            self.val_set = MyZT_dataset(val_list, transform, train_status=train_status,
                                        image_size=image_size, train_interval=train_interval,
                                        train_bins_interval=train_bins_interval, part1_train=part1_train,
                                        part2_train=part2_train, part3_train=part3_train,
                                        train_test_normal=train_test_normal, frac=frac, noise_level=noise_level,
                                          edge_lower=edge_lower, edge_up=edge_up,
                                          edge_lower_part3=edge_lower_part3, edge_up_part3=edge_up_part3,
                                        part_figure=int(part_figure*0.2), data_aug=data_aug)
        else:
            self.test_set = MyZT_dataset(root_list, transform, train_status=train_status,
                                        train_bins_interval=train_bins_interval, part3_test=part3_test,
                                         noise_level=noise_level,
                                          edge_lower=edge_lower, edge_up=edge_up,
                                          edge_lower_part3=edge_lower_part3, edge_up_part3=edge_up_part3)


class MyZT_dataset(Dataset):

    def __init__(self, root, transforms_ZT, image_size=128, train_status=True, test_interval=8,
                 train_bins_interval=8, part3_test=False, part1_train=False, part2_train=False,
                 part3_train=False, train_test_normal=False, frac=1, noise_level=3, edge_lower=15, edge_up=30,
                 edge_lower_part3=6, edge_up_part3=15, train_interval=128, data_aug=False, part_figure=42000):
        self.root = root
        self.transforms_ZT = transforms_ZT
        self.train = train_status
        self.image_size = image_size
        self.train_interval = train_interval
        self.test_interval = test_interval
        self.train_bins_interval = train_bins_interval
        self.bins = 256 // train_bins_interval
        self.part3_test = part3_test
        self.bins_dis = None
        self.look_up_dict = np.identity(self.bins)
        self.part1_train = part1_train
        self.part2_train = part2_train
        self.part3_train = part3_train
        self.train_test_normal = train_test_normal
        self.noise_level = noise_level
        self.edge_lower = edge_lower
        self.edge_up = edge_up
        self.edge_lower_part3 = edge_lower_part3
        self.edge_up_part3 = edge_up_part3
        self.frac = frac
        self.data_aug = data_aug
        self.part_figure = part_figure

        if train_status:
            self.list_image, self.list_generate_output = self.image_process()
        else:
            self.list_image = self.image_process()

    def __len__(self):
        return len(self.list_image)

    def __getitem__(self, index):
        """
        to ensure whether to padding
        """
        if self.train:
            ori_image = self.list_image[index]
            result_one_hot = self.list_generate_output[index]

            processed_ima = self.transforms_ZT(ori_image)
            return [processed_ima, result_one_hot]
        else:
            ori_image = self.list_image[index][0]
            processed_ima = self.transforms_ZT(ori_image)

            temp_tuple = self.list_image[index]
            temp_tuple[0] = processed_ima
            return temp_tuple

    def image_process(self):
        """
        list_image is a list of canny image, generate output is one hot encoder,
        correspond one by one
        """
        list_image = []
        logger = logging.getLogger()
        list_generate_output = []

        if self.train:
            limit_bias = min(self.generate_limit_bias(), 15)
            print("limit_bias:{}".format(limit_bias))

            round_len_root = max(round(self.frac * len(self.root)), 2)
            random_root_list = [self.root[i] for i in random.sample(range(0, len(self.root)), round_len_root)]

            slide_size = self.train_interval
            print("slide_size:{}".format(slide_size))
            for i in random_root_list:
                ori_image = cv2.imread(i)
                if self.part3_train:
                    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2LAB)

                init_num_row = random.randint(0, self.image_size)
                while (init_num_row + self.image_size) < ori_image.shape[0]:
                    init_num_col = random.randint(0, self.image_size)
                    while (init_num_col + self.image_size) < ori_image.shape[1]:
                        temp_ima = ori_image[init_num_row: init_num_row + self.image_size,
                                   init_num_col: init_num_col + self.image_size, :]
                        temp_ima_gray = temp_ima[:, :, 0]
                        blur = cv2.GaussianBlur(temp_ima_gray, (3, 3), self.noise_level)
                        mean_blur = np.sum(blur) / (blur.shape[0] * blur.shape[1])
                        if mean_blur < limit_bias:
                            init_num_col += slide_size
                            continue
                        else:
                            if not self.part3_train:
                                # repair: blur
                                canny = cv2.Canny(blur, self.edge_lower, self.edge_up)
                            else:
                                # if self.train_test_normal:
                                canny = cv2.Canny(temp_ima_gray, self.edge_lower_part3, self.edge_up_part3)
                            temp_ima = canny[:, :, np.newaxis]
                            list_image.append(temp_ima)
                            list_generate_output.append(self.generate_output(blur))
                            init_num_col += slide_size
                            # if self.part3_train:
                            if self.data_aug:
                                tt = random.randint(0, 10)
                                if tt >= 5:
                                    rand_crop_size = random.randint(90, 110)
                                    crop_ima = temp_ima_gray[:rand_crop_size, :rand_crop_size]
                                    temp_zero_ima = np.zeros((128, 128))
                                    temp_zero_ima[:rand_crop_size, :rand_crop_size] = crop_ima
                                    temp_zero_ima = temp_zero_ima.astype(np.uint8)

                                    crop_blur = cv2.GaussianBlur(temp_zero_ima, (3, 3), self.noise_level)
                                    if self.part3_train:
                                        crop_canny = cv2.Canny(temp_zero_ima, self.edge_lower_part3, self.edge_up_part3)
                                    else:
                                        crop_canny = cv2.Canny(crop_blur, self.edge_lower, self.edge_up)

                                    crop_temp_ima = crop_canny[:, :, np.newaxis]
                                    list_image.append(crop_temp_ima)
                                    list_generate_output.append(self.generate_output(crop_blur))
                    init_num_row += slide_size
                if len(list_image) > self.part_figure:
                    break
            logger.info('the number of all samples is {}'.format(len(list_image)))
            return list_image, list_generate_output
        else:
            for i in self.root:
                ori_image = cv2.imread(i)
                if self.part3_test:
                    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2LAB)
                ori_image = ori_image[:, :, 0]
                blur = cv2.GaussianBlur(ori_image, (3, 3), self.noise_level)
                if self.part3_test:
                    canny = cv2.Canny(ori_image, self.edge_lower_part3, self.edge_up_part3)
                else:
                    canny = cv2.Canny(blur, self.edge_lower, self.edge_up)
                if len(Counter(canny.reshape(-1))) == 1:
                    continue

                temp_ima = canny[:, :, np.newaxis]
                list_image.append([temp_ima, self.generate_output(blur), i])
            return list_image

    def generate_output(self, blur):
        blur_one_hot = blur / self.train_bins_interval
        blur_one_hot = np.floor(blur_one_hot).astype('int64')
        blur_one_hot = blur_one_hot.reshape(1, -1)
        return torch.tensor(blur_one_hot).long()

    def generate_limit_bias(self):
        mean_blur_list = []
        for i in self.root:
            ori_image = cv2.imread(i)
            if self.part3_train:
                ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2LAB)

            slide_size = self.train_interval

            init_num_row = 0
            while (init_num_row + self.image_size) < ori_image.shape[0]:
                init_num_col = 0
                while (init_num_col + self.image_size) < ori_image.shape[1]:
                    temp_ima = ori_image[init_num_row: init_num_row + self.image_size,
                               init_num_col: init_num_col + self.image_size, :]
                    temp_ima_gray = temp_ima[:, :, 0]
                    blur = cv2.GaussianBlur(temp_ima_gray, (3, 3), self.noise_level)
                    mean_blur = np.sum(blur) / (blur.shape[0] * blur.shape[1])
                    init_num_col += slide_size
                    mean_blur_list.append(mean_blur)
                init_num_row += slide_size

        aa = np.array(mean_blur_list)
        aaa = aa.reshape(-1)
        value_num, value_cor = np.histogram(aaa, bins=32)
        temp_value = np.concatenate((value_num[1:], np.array([0])))
        if len(np.where((temp_value - value_num) < -0.08 * np.sum(value_num))[0]):
            cor_index = np.where((temp_value - value_num) < -0.08 * np.sum(value_num))[0][0] + 1
        elif len(np.where((temp_value - value_num) < -0.03 * np.sum(value_num))[0]):
            cor_index = np.where((temp_value - value_num) < -0.03 * np.sum(value_num))[0][0] + 1
        else:
            cor_index = 1
        limit_bias = value_cor[cor_index]
        return limit_bias
