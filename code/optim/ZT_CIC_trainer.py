#!/usr/bin/python
# -*- coding:utf-8 -*-
from optim.base_trainer import BaseTrainer
from data_loader.base_load import BaseADDataset
from models.base import Base_Gray_paint_model
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
import time
import torch
import torch.optim as optim
import math
import random
import os
import pandas as pd

matplotlib.use('Agg')


def figure_dipict_save(array_numpy, filename):
    fig = plt.gcf()
    plt.imshow(array_numpy, cmap='gray')
    plt.axis("off")
    fig.savefig(filename, dpi=200)


class CICTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 64, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0, bins_interval=4):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Results
        self.net = None
        self.train_time = None
        self.test_time = None
        self.bins_interval = bins_interval

    def train(self, dataset: BaseADDataset, net: Base_Gray_paint_model,
              save_name='gray_colorization.pkl', Focal_loss=True, Focal_gama=2, show_figure=False,
              file_figure_name='./figure_store'):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, val_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        best_val_loss = None
        init_num = 0
        final_num = 0

        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            net.train()
            for data in train_loader:
                inputs, result_one_hot = data
                inputs = inputs.to(self.device)
                result_one_hot = result_one_hot.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
                outputs = net(inputs)
                sample_num = outputs.shape[0] * outputs.shape[2] * outputs.shape[3]
                if Focal_loss:
                    temp_outputs = (1 - outputs).pow(Focal_gama) * torch.log(outputs + 1e-5)
                    temp_outputs_view = temp_outputs.view(outputs.shape[0], outputs.shape[1], -1)
                    temp_outputs = torch.gather(temp_outputs_view, 1, result_one_hot)
                    loss = - torch.sum(temp_outputs) / sample_num
                else:
                    loss = - torch.sum(torch.log(outputs + 1e-5) * result_one_hot) / sample_num
                loss.backward()
                optimizer.step()

                loss_epoch += loss.item()
                print("loss_value is {}".format(loss.item()))
                n_batches += 1

            # log epoch statistics
            loss_epoch_test = 0.0
            n_batches_test = 0
            epoch_train_time = time.time() - epoch_start_time

            net.eval()
            for data in val_loader:
                inputs, result_one_hot = data
                inputs = inputs.to(self.device)
                result_one_hot = result_one_hot.to(self.device)

                with torch.no_grad():
                    outputs = net(inputs)
                    sample_num = outputs.shape[0] * outputs.shape[2] * outputs.shape[3]
                    if Focal_loss:
                        temp_outputs = (1 - outputs).pow(Focal_gama) * torch.log(outputs + 1e-5)
                        temp_outputs_view = temp_outputs.view(outputs.shape[0], outputs.shape[1], -1)
                        temp_outputs = torch.gather(temp_outputs_view, 1, result_one_hot)
                        loss = - torch.sum(temp_outputs) / sample_num
                    else:
                        loss = - torch.sum(torch.log(outputs + 1e-5) * result_one_hot) / sample_num

                    loss_epoch_test += loss.item()
                    n_batches_test += 1

            logger.info('Epoch {}/{}\t Time: {:.3f}\t Train_Loss: {:.8f}, Test_Loss: {:.8f}.'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches,
                                loss_epoch_test / n_batches_test))

            self.net = net

            temp_val_loss = loss_epoch_test / n_batches_test
            if not best_val_loss:
                best_val_loss = loss_epoch_test / n_batches_test
            else:
                if temp_val_loss < best_val_loss:
                    init_num = 0
                    best_val_loss = temp_val_loss
                    self.save_model(save_name=save_name)
                else:
                    init_num += 1

            if init_num > 6:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.1
                logger.info('current lr:{}'.format(optimizer.param_groups[0]['lr']))
                init_num = 0
                final_num += 1
            if final_num >= 2:
                break
            if math.isnan(temp_val_loss):
                logger.info('Val loss is nan')
                break

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)
        logger.info('Finished training.')

    def test(self, dataset: BaseADDataset, net: Base_Gray_paint_model,
             Focal_loss=True, Focal_gama=2, dip_fig=True, test_normal=False):
        logger = logging.getLogger()

        # Set device for network
        if test_normal:
            dict_tuple = {"loss": [], "std_loss": [], "pixel_loss_max": [],
                          "pixel_loss_max_square": []}
        else:
            dict_tuple = {"loss": [], "std_loss": [], "pixel_loss_max": [],
                          "pixel_loss_max_square": [], "root": []}
        net = net.to(self.device)
        self.net = net
        final_mediate_pixel = []
        final_mediate_pixel_prob = []
        final_mediate_pixel_std = []
        out_SR_list = []
        input_list = []

        # Get test data loader
        if not test_normal:
            test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
        else:
            test_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                if not test_normal:
                    inputs, result_one_hot, temp_root = data
                else:
                    inputs, result_one_hot = data
                inputs = inputs.to(self.device)
                result_one_hot = result_one_hot.to(self.device)
                temp_output = net(inputs)
                sample_num = temp_output.shape[2] * temp_output.shape[3]

                if Focal_loss:
                    temp_output_prob = 1 / (torch.max(temp_output, dim=1)[0] + 1e-5)
                    temp_output_std = 1 / (torch.std(temp_output, dim=1) + 1e-5)
                    temp_outputs = (1 - temp_output).pow(Focal_gama) * torch.log(temp_output + 1e-5)
                    temp_outputs_view = temp_outputs.view(temp_output.shape[0], temp_output.shape[1], -1)
                    temp_outputs = torch.gather(temp_outputs_view, 1, result_one_hot)
                    temp_outputs = temp_outputs.view(temp_outputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
                    inter_mediate_pixel = (- temp_outputs).squeeze(1)
                    loss = - torch.sum(temp_outputs.view(temp_outputs.shape[0], -1), dim=1) / sample_num

                loss = loss.cpu().detach()

                inter_mediate_pixel = inter_mediate_pixel.cpu().detach().numpy()
                if j == 0:
                    final_mediate_pixel = inter_mediate_pixel
                else:
                    final_mediate_pixel = np.concatenate((final_mediate_pixel, inter_mediate_pixel), axis=0)

                inter_temp_output_prob = temp_output_prob.cpu().detach().numpy()
                if j == 0:
                    final_mediate_pixel_prob = inter_temp_output_prob
                else:
                    final_mediate_pixel_prob = np.concatenate((final_mediate_pixel_prob, inter_temp_output_prob),
                                                              axis=0)

                inter_temp_output_std = temp_output_std.cpu().detach().numpy()
                if j == 0:
                    final_mediate_pixel_std = inter_temp_output_std
                else:
                    final_mediate_pixel_std = np.concatenate((final_mediate_pixel_std, inter_temp_output_std), axis=0)

                pixel_loss_list = inter_mediate_pixel.reshape(inter_mediate_pixel.shape[0], -1)
                pixel_loss_list_max = np.max(pixel_loss_list, axis=1)
                pixel_loss_list_max_square = np.max(
                    (pixel_loss_list - np.mean(pixel_loss_list, axis=1, keepdims=True)) ** 2, axis=1)
                std_loss = np.std(pixel_loss_list, 1)

                loss = list(loss.numpy())

                std_loss = list(std_loss)

                for ele_list in range(len(loss)):
                    dict_tuple["loss"].append(loss[ele_list])
                    dict_tuple["std_loss"].append(std_loss[ele_list])

                    dict_tuple["pixel_loss_max"].append(pixel_loss_list_max[ele_list])
                    dict_tuple["pixel_loss_max_square"].append(pixel_loss_list_max_square[ele_list])

                    if not test_normal:
                        dict_tuple["root"].append(temp_root[ele_list].split("/")[-1].split(".")[0])

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        logger.info('Finished testing.')

        pd_test = pd.DataFrame(dict_tuple)
        if dip_fig:
            return pd_test, input_list, out_SR_list, final_mediate_pixel, \
                   final_mediate_pixel_prob, final_mediate_pixel_std
        else:
            return pd_test

    def save_model(self, save_path='', save_name='part1'):
        torch.save(self.net.state_dict(), save_path + save_name)

    def generate_downsample_SR(self, single_output):
        # single_output: channel x H x W (tensor) ----> H x W (array)
        temp_output = torch.argmax(single_output, dim=0)
        temp_output = (temp_output * self.bins_interval + self.bins_interval / 2)
        # temp_SR_output = F.upsample(temp_output.unsqueeze(0).unsqueeze(0), scale_factor=4,
        #                             mode='bilinear')

        return temp_output.numpy()#, temp_SR_output.squeeze(0).squeeze(0).numpy()

    def generate_output(self, temp_ima, filename):
        temp_ima_cuda = temp_ima.unsqueeze(0).unsqueeze(0).to(self.device)
        self.net.eval()
        with torch.no_grad():
            temp_output = self.net(temp_ima_cuda)
            temp_output = torch.argmax(temp_output.cpu().detach()[0], dim=0)
            temp_output = (temp_output * self.bins_interval + self.bins_interval / 2)
            final_output = temp_output.numpy()
            figure_dipict_save(final_output, filename)

    def random_list_figure_generate(self, dataloder, temp_random_list,
                                    input_epoch, mode='train', file_figure_name='./figure_store'):
        for figure_index in temp_random_list:
            temp_ima = list(dataloder.dataset)[figure_index][0][0]
            temp_canny = temp_ima.numpy()
            temp_file_name = file_figure_name + '/epoch_'
            if not os.path.exists(temp_file_name + str(input_epoch)):
                os.mkdir(temp_file_name + str(input_epoch))

            if mode == 'train':
                common_name = '/train_epoch_'
            else:
                common_name = '/val_epoch_'
            temp_name = temp_file_name + str(input_epoch) + common_name + str(input_epoch) + '_index_' \
                        + str(figure_index) + '_canny' + '.png'

            temp_name2 = temp_file_name + str(input_epoch) + common_name + str(input_epoch) + '_index_' \
                         + str(figure_index) + '_generate_figure' + '.png'

            figure_dipict_save(temp_canny, temp_name)
            self.generate_output(temp_ima, temp_name2)
