#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import zipfile
import cv2
from glob import glob
import numpy as np
from collections import Counter
import json
import argparse
import gc
import torch
from data_loader.ZT_datasets import ZT_Dataset
from models.ZT_U_renet import Generator
from optim.ZT_CIC_trainer import CICTrainer
import math
import time
import logging

import warnings
warnings.filterwarnings("ignore")

train_path1 = "../raw_data/round_train/part1/OK_Images/"
train_path2 = "../raw_data/round_train/part2/OK_Images/"
train_path3 = "../raw_data/round_train/part3/OK_Images/"
test_path1 = "../raw_data/round_test/part1/TC_Images/"
test_path2 = "../raw_data/round_test/part2/TC_Images/"
test_path3 = "../raw_data/round_test/part3/TC_Images/"
model_path = '../model/'
model_path1 = '../model/CIC_part1.pkl'
model_path2 = '../model/CIC_part2.pkl'
model_path3 = '../model/CIC_part3.pkl'
temp_path = '../temp_data/'
save_path1 = '../temp_data/result/data/focusight1_round2_train_part1/TC_Images/'
save_path2 = '../temp_data/result/data/focusight1_round2_train_part2/TC_Images/'
save_path3 = '../temp_data/result/data/focusight1_round2_train_part3/TC_Images/'
result_path = '../result/'
# repair tjiao
slide_dis = 1
# repair tijiao
frac = 1
part_figure1 = 42000
part_figure2 = 42000
part_figure3 = 52000


def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lr", default=0.0001,
                        type=float, help="up bound of canny threshold of part2")
    # revise tijiao
    parser.add_argument("--train_epoch", default=100,
                        type=int, help="low bound of canny threshold of part2")
    parser.add_argument("--train_batch_size", default=72,
                        type=int, help="the level of edge denoising of part2")
    parser.add_argument("--test_batch_size", default=256,
                        type=int, help="the level of edge denoising of part2")
    parser.add_argument("--bins_interval", default=8,
                        type=int, help="the mode of edge denoising of part2")
    parser.add_argument("--save_path", default='',
                        type=str, help="the error to distinguish the state in part2")
    parser.add_argument("--save_name", default='part1',
                        type=str, help="the error to distinguish the state in part2")
    parser.add_argument("--file_name", default='./figure_store',
                        type=str, help="the error to distinguish the state in part2")

    config = parser.parse_args()
    return config


def test_model(root, bins_interval, models_root, batch_size, device, part3_test=False, senet=False):
    gray_paint = Generator(bins=(int(256 // bins_interval)), senet=senet)
    gray_paint.load_state_dict(torch.load(models_root))

    dataset = ZT_Dataset(root=root, train_bins_interval=bins_interval, train_status=False, part3_test=part3_test)
    trainer = CICTrainer(batch_size=batch_size, device=device, bins_interval=bins_interval)

    pd_test, input_list, out_SR_list, out_list, final_mediate_pixel_prob, final_mediate_pixel_std = trainer.test(
        net=gray_paint, dataset=dataset)
    return pd_test, input_list, out_SR_list, out_list, final_mediate_pixel_prob, final_mediate_pixel_std


def generate_abnormal_root(pd_test, threshold_down, feature_name):
    norm_std_loss_csv = pd_test.sort_values(by=feature_name)
    total_root = list(pd_test['root'])

    temp_std_root = list(norm_std_loss_csv[norm_std_loss_csv[feature_name] < threshold_down]['root'])
    abnormal_list = [i for i in total_root if i not in temp_std_root]

    list_loss_csv = []

    for i in abnormal_list:
        temp_index, temp_rt = pd_test[pd_test['root'] == i].index[0], \
                              pd_test[pd_test['root'] == i]['root'].values[0]
        list_loss_csv.append((temp_index, temp_rt))
    return list_loss_csv


def generate_json_dict(propose_path, temp_tuple):
    dict_temp = {'Height': 128, 'Width': 128, 'name': propose_path + '.bmp',
                 'regions': []}
    temp_dict = {'points': []}
    for i in range(len(temp_tuple[0])):
        temp_dict['points'].append('{}, {}'.format(temp_tuple[0][i], temp_tuple[1][i]))
    dict_temp['regions'].append(temp_dict)
    return dict_temp


def crop_ima(ima, crop_size):
    ima[0: crop_size, :] = 0
    ima[(ima.shape[0] - crop_size):, :] = 0
    ima[:, 0: crop_size] = 0
    ima[:, (ima.shape[0] - crop_size):] = 0
    return ima


def generate_canny(fig, frac=11, mode='1', fig2=None, frac_2=None, add_path=None,
                   crop_size_1=0, crop_size_2=28, crop_size_3=28,
                   limit_small=50, limit_all=100, crop_size_all=8, crop_size_small=40):
    if mode == '1':
        a = np.zeros((fig.shape[0], fig.shape[0]))
        final_fig = (fig - np.mean(fig)) ** 2
        a[final_fig > frac] = 1
        a = crop_ima(a, crop_size_1)
        temp_position = np.where(a == 1)
    elif mode == '2':
        a = np.zeros((fig.shape[0], fig.shape[0]))
        final_fig = (fig - np.mean(fig)) ** 2
        a[final_fig > frac] = 1
        a = crop_ima(a, crop_size_2)
        temp_position = np.where(a == 1)
    elif mode == '3':
        a = np.zeros((fig.shape[0], fig.shape[0]))
        final_fig = (fig - np.mean(fig)) ** 2
        cc = np.sort(final_fig.reshape(-1))
        threshold = cc[-math.floor(0.035 * len(cc))]
        a[final_fig > threshold] = 1
        a = crop_ima(a, crop_size_3)
        temp_position = np.where(a == 1)
        current_path = add_path
        _, temp_canny = image_edge_preprocess(current_path, threshold_up=20, threshold_down=10, denoise_level=3)
        aa = np.zeros((fig.shape[0], fig.shape[0]))
        aa[temp_canny == 255] = 1
        aa_1 = crop_ima(aa, crop_size_all)
        aa_2 = crop_ima(aa, crop_size_small)
        if len(np.where(aa_2 == 1)[0]) < limit_small:
            temp_canny = dilate_oper(temp_canny, kernel_size=3)
            aaa = np.zeros((fig.shape[0], fig.shape[0]))
            aaa[temp_canny == 255] = 1
            aaa = crop_ima(aaa, crop_size_small)
            temp_position = np.where(aaa == 1)
        elif len(np.where(aa_1 == 1)[0]) < limit_all:
            temp_canny = dilate_oper(temp_canny, kernel_size=3)
            aaa = np.zeros((fig.shape[0], fig.shape[0]))
            aaa[temp_canny == 255] = 1
            temp_position = np.where(aaa == 1)
    return temp_position


def generate_json_result(abnormal_list, out_array_part1, save_path, frac=11, mode='1',
                         out_array_part2_std=None, frac_2=None, default_path=None, cr_limit=-20,
                         cr_limit_sum=20, fine_tune=10, fine_tune_limit_pixel=80, cr_size=8):
    for i in abnormal_list:
        temp_index, temp_root = i[0], i[1]
        if mode == '1':
            temp_position = generate_canny(fig=out_array_part1[temp_index], frac=frac, mode=mode)
        elif mode == '2':
            temp_position = generate_canny(fig=out_array_part1[temp_index], frac=frac, mode=mode,
                                           fig2=out_array_part2_std[temp_index], frac_2=frac_2)
        elif mode == "3":
            add_path = default_path + temp_root + '.bmp'
            temp_position = generate_canny(fig=out_array_part1[temp_index], frac=frac, mode=mode, add_path=add_path)
            temp_ima, temp_canny = image_edge_preprocess(add_path, denoise_level=3)
            if np.sum(temp_ima[cr_limit:, :]) < cr_limit_sum and np.sum(temp_ima[:, cr_limit:]) < cr_limit_sum:
                tt = temp_ima[0, :]
                ttt = np.concatenate((np.array([0]), tt[:-1]))
                non_zero_position = np.where((tt - ttt) != 0)[0][-1]
                if len(np.where(temp_canny[:non_zero_position - fine_tune, :non_zero_position - fine_tune] == 255)[0]) == 0:
                    continue
                elif len(np.where(temp_canny[:non_zero_position - fine_tune, :non_zero_position - fine_tune] == 255)[0]) < fine_tune_limit_pixel:
                    temp_canny = dilate_oper(temp_canny)
                    temp_position = np.where(temp_canny[:non_zero_position - fine_tune, :non_zero_position - fine_tune] == 255)
                else:
                    temp_position = np.where(temp_canny[:non_zero_position - fine_tune, :non_zero_position - fine_tune] == 255)
        if len(temp_position[0]) == 0:
            current_path = default_path + temp_root + '.bmp'
            _, temp_canny = image_edge_preprocess(current_path, threshold_up=20, threshold_down=10, denoise_level=3)
            temp_canny = dilate_oper(temp_canny)
            a = np.zeros((temp_canny.shape[0], temp_canny.shape[0]))
            a[temp_canny == 255] = 1
            a = crop_ima(a, cr_size)
            temp_position = np.where(a == 1)
            if len(temp_position[0]) == 0:
                continue
        temp_dict = generate_json_dict(temp_root, temp_position)

        savefile = save_path + temp_root + '.json'
        with open(savefile, 'w') as f:
            f.write(json.dumps(temp_dict, ensure_ascii=False, indent=2))


def zipdir(path, file):
    z = zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(path):
        fpath = dirpath.replace(path, '')
        fpath = fpath and fpath + os.sep or ''
        for filename in filenames:
            z.write(os.path.join(dirpath, filename), fpath+filename)
    z.close()


def generate_json_dict_part3(propose_path, temp_tuple):
    dict_temp = {'Height': 128, 'Width': 128, 'name': propose_path.split('/')[-1],
                 'regions': []}
    temp_dict = {'points': []}
    for i in range(len(temp_tuple[0])):
        temp_dict['points'].append('{}, {}'.format(temp_tuple[0][i], temp_tuple[1][i]))
    dict_temp['regions'].append(temp_dict)
    return dict_temp


def generate_root(path):
    train_list = glob(os.path.join(path, '*.bmp'))
    return train_list


def image_edge_preprocess(propose_path, threshold_up=20, threshold_down=10, denoise_mode='1', denoise_level=3):
    # part2: noise_level:5, 120,150
    img = cv2.imread(propose_path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if denoise_mode == '2':
        blur = cv2.medianBlur(img, denoise_level)
    elif denoise_mode == '1':
        blur = cv2.GaussianBlur(img, (3, 3), denoise_level)
    canny = cv2.Canny(blur, threshold_down, threshold_up)
    return img, canny


def filter_image(propose_path, threshold_up=20, threshold_down=10, denoise_mode='1', denoise_level=3):
    filter_path = []
    for path in propose_path:
        temp_count = Counter(image_edge_preprocess(path, threshold_up, threshold_down, denoise_mode, denoise_level)[1].flatten())
        if len(temp_count) == 2:
            filter_path.append(path)
    return filter_path


def temp_part3_generate_result(pro_path, mode='1'):
    image, canny = image_edge_preprocess(pro_path, 20, 10, '1', 3)
    canny = dilate_oper(canny)
    if mode == '1':
        temp_tuple = np.where(canny == 255)
    else:
        tt = image[0, :]
        ttt = np.concatenate((np.array([0]), tt[:-1]))
        non_zero_position = np.where((tt - ttt) != 0)[0][-1]

        temp_tuple = np.where(canny[:non_zero_position-10, :non_zero_position-10] == 255)
    final_dict = generate_json_dict_part3(pro_path, temp_tuple)
    return final_dict


def part3_generate_result(pro_path, save_path, mode='1'):
    for path in pro_path:
        temp_dict = temp_part3_generate_result(path, mode=mode)
        temp_split = path.split('/')
        image = temp_split[-1].split('.')[0]
        savefile = save_path + image + '.json'
        with open(savefile, 'w') as f:
            f.write(json.dumps(temp_dict, ensure_ascii=False, indent=2))


def dilate_oper(canny, kernel_size=3, iter_num=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilate = cv2.dilate(canny, kernel, iter_num)
    return dilate


def train_model(net, trainer, dataset, batch_size, bins_interval, part_ok, save_name,
                figure_save_name, device, lr, show_figure=False, part3_train=False, slide_dis=3
                , part1_train=False, part2_train=False, frac=1, load_model=False, senet=True, epochs=100,
                part_figure=42000, test_normal=False, data_aug=False, ratio_val_max=75, ratio_val_square=75):
    gray_paint = net(bins=(int(256 // bins_interval)), senet=senet)
    if load_model:
        gray_paint.load_state_dict(torch.load(save_name))

    if test_normal:
        gray_paint.load_state_dict(torch.load(save_name))
    BetaVAETrainer_CIC = trainer(batch_size=batch_size, bins_interval=bins_interval, device=device, lr=lr,
                                 n_epochs=epochs)

    # Train model on dataset
    if not test_normal:
        dataset = dataset(root=part_ok, train_bins_interval=bins_interval, part3_train=part3_train,
                          part2_train=part2_train, part1_train=part1_train, frac=frac, part_figure=part_figure,
                          data_aug=data_aug)
        BetaVAETrainer_CIC.train(dataset=dataset, net=gray_paint,
                                 save_name=save_name, file_figure_name=figure_save_name, show_figure=show_figure)
    else:
        dataset = dataset(root=part_ok, train_bins_interval=bins_interval, part3_train=part3_train,
                          part2_train=part2_train, part1_train=part1_train, frac=frac, part_figure=part_figure,
                          data_aug=data_aug)
        pd_test, _, _, _, _, _ = BetaVAETrainer_CIC.test(dataset=dataset, net=gray_paint, test_normal=test_normal)
        res_list = pd_test['pixel_loss_max'].values
        percentiles = np.array([2.5, 40, 50, ratio_val_max, 97.5])
        ptiles_vers = np.percentile(res_list, percentiles)
        pixel_loss_max_index = ptiles_vers[3]
        res_list = pd_test['pixel_loss_max_square'].values
        percentiles = np.array([2.5, 40, 50, ratio_val_square, 97.5])
        ptiles_vers = np.percentile(res_list, percentiles)
        pixel_loss_max_square_index = ptiles_vers[3]
        return pixel_loss_max_index, pixel_loss_max_square_index


def main():

    # Get configuration
    cfg = get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device("cuda:0")

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    now = int(time.time())
    timeArray = time.localtime(now)
    otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
    otherStyleTime = otherStyleTime.replace(" ", "_").replace("--", "_").replace(":", "_")
    log_file = './checkpoints' + '/log_' + otherStyleTime + '_CIC_' + cfg.save_name + '.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console)

    # Print arguments
    logger.info('Log file is %s.' % log_file)

    # Log training details
    logger.info('Training learning rate: %g' % cfg.lr)
    logger.info('Training epochs: %d' % cfg.train_epoch)
    logger.info('Training batch size: %d' % cfg.train_batch_size)

    logger.info('start part1 training')
    train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset, batch_size=cfg.train_batch_size,
                bins_interval=cfg.bins_interval, part_ok=train_path1, save_name=model_path1,
                figure_save_name=cfg.file_name, device=device, lr=cfg.lr, slide_dis=slide_dis, part1_train=True,
                frac=frac, senet=True, epochs=cfg.train_epoch, part_figure=part_figure1, data_aug=False)
    gc.collect()

    logger.info('start part2 training')
    train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset, batch_size=cfg.train_batch_size,
                bins_interval=cfg.bins_interval, part_ok=train_path2, save_name=model_path2,
                figure_save_name=cfg.file_name, device=device, lr=cfg.lr, slide_dis=slide_dis, part2_train=True,
                frac=frac, senet=True, epochs=cfg.train_epoch, part_figure=part_figure2, data_aug=False)
    gc.collect()

    logger.info('start part3 training')
    train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset, batch_size=cfg.train_batch_size,
                bins_interval=cfg.bins_interval, part_ok=train_path3, save_name=model_path3,
                figure_save_name=cfg.file_name, device=device, part3_train=True, lr=cfg.lr, slide_dis=slide_dis,
                frac=frac, senet=True, epochs=cfg.train_epoch, part_figure=part_figure3, data_aug=True)
    gc.collect()

    pixel_max_1, pixel_max_square_1 = train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset,
                                                  batch_size=cfg.test_batch_size,
                                                  bins_interval=cfg.bins_interval, part_ok=train_path1,
                                                  save_name=model_path1,
                                                  figure_save_name=cfg.file_name, device=device, lr=cfg.lr,
                                                  slide_dis=slide_dis, part1_train=True,
                                                  frac=frac, senet=True, epochs=cfg.train_epoch,
                                                  part_figure=part_figure1, test_normal=True, data_aug=False)

    pixel_max_2, pixel_max_square_2 = train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset,
                                                  batch_size=cfg.test_batch_size,
                                                  bins_interval=cfg.bins_interval, part_ok=train_path2,
                                                  save_name=model_path2,
                                                  figure_save_name=cfg.file_name, device=device, lr=cfg.lr,
                                                  slide_dis=slide_dis, part2_train=True,
                                                  frac=frac, senet=True, epochs=cfg.train_epoch,
                                                  part_figure=part_figure2, test_normal=True, data_aug=False,
                                                  ratio_val_square=85)

    pixel_max_3, pixel_max_square_3 = train_model(net=Generator, trainer=CICTrainer, dataset=ZT_Dataset,
                                                  batch_size=cfg.test_batch_size,
                                                  bins_interval=cfg.bins_interval, part_ok=train_path3,
                                                  save_name=model_path3,
                                                  figure_save_name=cfg.file_name, device=device, lr=cfg.lr,
                                                  slide_dis=slide_dis, part3_train=True,
                                                  frac=frac, senet=True, epochs=cfg.train_epoch,
                                                  part_figure=part_figure3, test_normal=True, data_aug=False,
                                                  ratio_val_square=35, ratio_val_max=35)
    logger.info('start test')

    # # root1
    logger.info('start test part1')
    pd_test_part1, _, _, out_array_part1, _, _ = test_model(root=test_path1, bins_interval=cfg.bins_interval,
                                                            models_root=model_path1,
                                                            batch_size=cfg.test_batch_size, device=device, senet=True)
    abnormal_root1 = generate_abnormal_root(pd_test_part1, pixel_max_1, feature_name='pixel_loss_max')
    generate_json_result(abnormal_root1, out_array_part1, save_path1, frac=pixel_max_square_1, default_path=test_path1)

    del pd_test_part1
    del out_array_part1
    gc.collect()
    logger.info('start test part2')
    #
    # # root2
    pd_test_part2, _, _, out_array_part2, _, out_array_part2_std = test_model(root=test_path2,
                                                                              bins_interval=cfg.bins_interval,
                                                                              models_root=model_path2,
                                                                              batch_size=cfg.test_batch_size,
                                                                              device=device, senet=True)
    abnormal_root2 = generate_abnormal_root(pd_test_part2, pixel_max_2, feature_name='pixel_loss_max')
    generate_json_result(abnormal_root2, out_array_part2, save_path2, frac=pixel_max_square_2, mode='2',
                         out_array_part2_std=out_array_part2_std, frac_2=0.04, default_path=test_path2)

    del pd_test_part2
    del out_array_part2
    del out_array_part2_std
    gc.collect()
    logger.info('start test part3')

    # root3
    pd_test_part3, _, _, out_array_part3, _, _ = test_model(root=test_path3,
                                                            bins_interval=cfg.bins_interval,
                                                            models_root=model_path3,
                                                            batch_size=cfg.test_batch_size,
                                                            device=device, part3_test=True, senet=True)
    abnormal_root3 = generate_abnormal_root(pd_test_part3, pixel_max_3, feature_name='pixel_loss_max')
    generate_json_result(abnormal_root3, out_array_part3, save_path3, frac=pixel_max_square_3, mode='3',
                         out_array_part2_std=None, default_path=test_path3)

    list_extra_fault = []
    list_extra_fault_2 = []
    root3_path = generate_root(test_path3)
    filter_path = filter_image(root3_path)
    normal_path = [i for i in root3_path if i not in filter_path]
    for i in normal_path:
        temp_split = i.split('/')
        image = temp_split[-1].split('.')[0]
        savefile = save_path3 + image + '.json'
        if os.path.exists(savefile):
            os.remove(savefile)

    for i in filter_path:
        temp_ima, temp_canny = image_edge_preprocess(i, denoise_level=3)
        if len(np.where(temp_canny == 255)[0]) < 100:
            list_extra_fault.append(i)
    part3_generate_result(list_extra_fault, save_path3)

    zipdir(temp_path + 'result/', result_path + 'data.zip')
    print("zip successfully!")


if __name__ == "__main__":
    main()
