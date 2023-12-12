# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from models.models import create_model
from models.LRecModel import LRecModel
import util.util as util
import PIL.Image as img
from PIL import Image
# import pytorch_ssim
import torch
# import torchvision.utils as vutils
import torchvision.transforms as transforms
import numpy as np
# from skimage.measure import compare_ssim
# from skimage.measure import compare_psnr
# from skimage.metrics import structural_similarity
import cv2
import glob as gb
from os.path import join
from tqdm import tqdm
# import random

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def event2frame(event, img_size, ts, f_span, num_frame):
    ## convert event streams to [T, C, H, W] event tensor, C=2 indicates polarity
    f_start, f_end = f_span
    preE = np.zeros((num_frame, 2, img_size[0], img_size[1]))
    postE = np.zeros((num_frame, 2, img_size[0], img_size[1]))
    interval = (f_end - f_start) / num_frame  # based on whole event range

    if event['t'].shape[0] > 0:
        preE = e2f_detail(event, preE, ts, f_start, interval)
        postE = e2f_detail(event, postE, ts, f_end, interval)
    return preE, postE

def filter_events(event_data, start, end):
    ## filter events based on temporal dimension
    x = event_data['x'][event_data['t'] >= start]
    y = event_data['y'][event_data['t'] >= start]
    p = event_data['p'][event_data['t'] >= start]
    t = event_data['t'][event_data['t'] >= start]

    x = x[t <= end]
    y = y[t <= end]
    p = p[t <= end]
    t = t[t <= end]
    return x, y, p, t

def e2f_detail(event_in, eframe, ts, key_t, interval):
    event = event_in
    T, C, H, W = eframe.shape
    eframe = eframe.ravel()
    if key_t < ts:
        ## reverse event time & porlarity
        x, y, p, t = filter_events(event, key_t, ts)  # filter events by time
        new_t = ts - t
        idx = np.floor(new_t / interval).astype(int)
        idx[idx == T] -= 1
        # assert(idx.max()<T)
        p[p == -1] = 0  # reversed porlarity
        np.add.at(eframe, x + y * W + p * W * H + idx * W * H * C, 1)
    else:
        x, y, p, t = filter_events(event, ts, key_t)  # filter events by time
        new_t = t - ts
        idx = np.floor(new_t / interval).astype(int)
        idx[idx == T] -= 1
        # assert(idx.max()<T)
        p[p == 1] = 0  # pos in channel 0
        p[p == -1] = 1  # neg in channel 1
        np.add.at(eframe, x + y * W + p * W * H + idx * W * H * C, 1)
    eframe = np.reshape(eframe, (T, C, H, W))
    return eframe

def fold_time_dim(inp):
    if inp.ndim == 4:
        T, C, H, W = inp.shape
        out = inp.reshape((T * C, H, W))  # [T,C,H,W] -> [T*C,H,W]
    elif inp.ndim == 5:
        N, T, C, H, W = inp.shape
        out = inp.reshape((N, T * C, H, W))  # [N,T,C,H,W] -> [N,T*C,H,W]
    return out

def process_event(temp_event, ts, trigger):
    if len(temp_event.shape) > 1:
        temp_event_data = {'t': temp_event[:, 0], 'x': temp_event[:, 1].astype(np.int32), 'y': temp_event[:, 2].astype(np.int32),
                            'p': temp_event[:, 3].astype(np.int32)}

        temp_event_data['p'][temp_event_data['p'] == 0] = -1  # neg in channel 1
        # print('temp_event_data_t', temp_event_data['t'])
        # print('temp_event_data_x', temp_event_data['x'])
        # print('temp_event_data_y', temp_event_data['y'])
        # print('temp_event_data_p', temp_event_data['p'])
        img_size = (256, 256)
        leftB_tmp, rightB_tmp = event2frame(temp_event_data, img_size, ts, trigger, 13)
        _, B_all_start = event2frame(temp_event_data, img_size, trigger[0], trigger, 13)
        leftB_tmp = fold_time_dim(leftB_tmp)
        rightB_tmp = fold_time_dim(rightB_tmp)
        B_all_start = fold_time_dim(B_all_start)
        assert (np.max(leftB_tmp) < 255)
        assert (np.max(rightB_tmp) < 255)
        leftB_tmp = np.array(leftB_tmp, dtype=np.int8)
        rightB_tmp = np.array(rightB_tmp, dtype=np.int8)
    return leftB_tmp, rightB_tmp, B_all_start

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    opt.checkpoints_dir = "./checkpoints_SYN/"

if __name__ == "__main__":

    opt = TestOptions().parse(save=False)
    parameter_set(opt)
    print("*************************************pth:", opt.which_epoch)
    model = LRecModel()
    model.initialize(opt)
    model.eval()
    dataset_size = 0
    dataroot = opt.test_input
    outputs_dir = opt.outputs_dir
    transform_list = []
    transform_list.append(transforms.ToTensor())
    transform_img = transforms.Compose(transform_list)
    tonemapReinhard = cv2.createTonemapReinhard(1.5, 0,0,0)
    LDRBpth = dataroot
    image_name_s = gb.glob(os.path.join(LDRBpth, '*/LDR_gamma/*.png'))
    # print(dataroot)
    for image_name in tqdm(image_name_s):
        output_scene = image_name.split('/')[-3]
        img_number = image_name.split('/')[-1].split('.')[0]

        LDRB_in = cv2.imread(image_name)
        LDRB = LDRB_in[:, :, [2, 1, 0]]
        LDRB = transform_img(LDRB).unsqueeze(0)

        temp_event = np.load(image_name.replace('LDR_gamma','event_raw').replace('.png','.npz'))
        events = temp_event['events']
        trigger = temp_event['trigger']

        number_frame = 13
        for i in range(number_frame):
            event_leftB, event_rightB, B_all_start = process_event(events, trigger[i], (temp_event['trigger'][0], temp_event['trigger'][-1]))
            event_leftB = torch.from_numpy(event_leftB).unsqueeze(0)
            event_rightB = torch.from_numpy(event_rightB).unsqueeze(0)
            B_all_start = torch.from_numpy(B_all_start).unsqueeze(0)
            
            fake_HDR_S = model.inference(LDRB, event_leftB, event_rightB, B_all_start)

            fake_HDR_S = fake_HDR_S.data.cpu().numpy()
            fake_HDR_S = fake_HDR_S[0, ...].transpose(1, 2, 0)
            fake_HDR_S = fake_HDR_S[:, :, ::-1]

            fake_HDR_S = np.interp(fake_HDR_S, [fake_HDR_S.min(), fake_HDR_S.max()], [0, 1]).astype(fake_HDR_S.dtype)
            # gene_nump = (gene_nump.clip(0, 1))
            # gene_nump = tonemapReinhard.process(gene_nump)
            # gene_nump = (gene_nump.clip(0, 1) * 255).astype(np.uint8)
            ensure_dir(join(outputs_dir, 'synthetic', output_scene.zfill(5), str(img_number).zfill(5)))
            cv2.imwrite(join(outputs_dir, 'synthetic', output_scene.zfill(5), str(img_number).zfill(5), str(i).zfill(2) + '.hdr'), fake_HDR_S)
            
            # fake_HDR_Recon = fake_HDR_Recon.data.cpu().numpy()
            # fake_HDR_Recon = fake_HDR_Recon[0, ...].transpose(1, 2, 0)
            # fake_HDR_Recon = fake_HDR_Recon[:, :, ::-1]
            # fake_HDR_Recon = np.interp(fake_HDR_Recon, [fake_HDR_Recon.min(), fake_HDR_Recon.max()], [0, 1]).astype(fake_HDR_Recon.dtype)
            # # gene_nump = (gene_nump.clip(0, 1))
            # # gene_nump = tonemapReinhard.process(gene_nump)
            # # gene_nump = (gene_nump.clip(0, 1) * 255).astype(np.uint8)
            # ensure_dir(join(outputs_dir.replace('ours/','ours_all/'), 'synthetic', output_scene.zfill(5), str(img_number).zfill(5)))
            # cv2.imwrite(join(outputs_dir.replace('ours/','ours_all/'), 'synthetic', output_scene.zfill(5), str(img_number).zfill(5), str(i).zfill(2) + '.hdr'), fake_HDR_Recon)
