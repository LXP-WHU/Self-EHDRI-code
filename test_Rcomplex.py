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
        temp_event_data = {'t': temp_event[:, 0], 'x': temp_event[:, 1], 'y': temp_event[:, 2],
                            'p': temp_event[:, 3]}
        temp_event_data['p'][temp_event_data['p'] == 0] = -1  # neg in channel 1
        img_size = (256, 256)
        time_in = trigger[0]+ts*(trigger[1]-trigger[0])
        leftB_tmp, rightB_tmp = event2frame(temp_event_data, img_size, time_in, trigger, 13)
        _, B_all_start = event2frame(temp_event_data, img_size, trigger[0], trigger, 13)
        B_all_end, _ = event2frame(temp_event_data, img_size, trigger[1], trigger, 13)
        leftB_tmp = fold_time_dim(leftB_tmp)
        rightB_tmp = fold_time_dim(rightB_tmp)
        B_all_start = fold_time_dim(B_all_start)
        B_all_end = fold_time_dim(B_all_end)
        assert (np.max(leftB_tmp) < 255)
        assert (np.max(rightB_tmp) < 255)
        leftB_tmp = np.array(leftB_tmp, dtype=np.int8)
        rightB_tmp = np.array(rightB_tmp, dtype=np.int8)
    return leftB_tmp, rightB_tmp, B_all_start, B_all_end

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def parameter_set(opt):
    ## Default parameters
    opt.serial_batches = True  # no shuffle
    # opt.checkpoints_dir = "./checkpoints/"

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
    LDRBpth = os.path.join(dataroot, 'dynamic')
    scenes_lists = gb.glob(os.path.join(LDRBpth, '91'))
    # print(dataroot)
    for scene_name in scenes_lists:
        # print('scene_name',scene_name)
        scene_number = scene_name.split('dynamic/')[-1]
        trigger_name = os.path.join(scene_name, 'trigger.npy')
        trigger_all = np.load(trigger_name)
        for i in tqdm(range(48,49,1)):
            image_name = os.path.join(scene_name, 'LDR_gamma', str(i).zfill(5)+'.png')
            event_name = os.path.join(scene_name, 'event_raw', str(i).zfill(5)+'.npy')
            
            LDRB_in = cv2.imread(image_name)
            LDRB = LDRB_in[:, :, [2, 1, 0]]
            LDRB = transform_img(LDRB).unsqueeze(0)

            temp_event = np.load(event_name)
            trigger = (trigger_all[2*i][1], trigger_all[2*i+1][1])
            number_frame = 90
            timestamps = np.linspace(0, 1, 90, dtype=np.float32)
            for time_num in range(number_frame):
                time_dynamic = timestamps[time_num]
                # print(time_dynamic)
                event_leftB, event_rightB, B_all_start, B_all_end = process_event(temp_event, time_dynamic, trigger)
                event_leftB = torch.from_numpy(event_leftB).unsqueeze(0)
                event_rightB = torch.from_numpy(event_rightB).unsqueeze(0)
                B_all_start = torch.from_numpy(B_all_start).unsqueeze(0)
                B_all_end = torch.from_numpy(B_all_end).unsqueeze(0)
                fake_HDR_S, fake_LDR_S, fake_HDR_Recon = model.inference(LDRB, event_leftB, event_rightB, B_all_start)

                fake_HDR_S = fake_HDR_S.data.cpu().numpy()
                fake_HDR_S = fake_HDR_S[0, ...].transpose(1, 2, 0)
                fake_HDR_S = fake_HDR_S[:, :, ::-1]
                fake_HDR_S = np.interp(fake_HDR_S, [fake_HDR_S.min(), fake_HDR_S.max()], [0, 1]).astype(fake_HDR_S.dtype)
                # gene_nump = (gene_nump.clip(0, 1))
                # gene_nump = tonemapReinhard.process(gene_nump)
                # gene_nump = (gene_nump.clip(0, 1) * 255).astype(np.uint8)
                ensure_dir(join(outputs_dir, 'dynamic', scene_number, str(i).zfill(5)))
                cv2.imwrite(join(outputs_dir, 'dynamic', scene_number, str(i).zfill(5), str(time_num).zfill(2) + '.hdr'), fake_HDR_S)
             
            #     fake_HDR_Recon = fake_HDR_Recon.data.cpu().numpy()
            #     fake_HDR_Recon = fake_HDR_Recon[0, ...].transpose(1, 2, 0)
            #     fake_HDR_Recon = fake_HDR_Recon[:, :, ::-1]
            #     fake_HDR_Recon = np.interp(fake_HDR_Recon, [fake_HDR_Recon.min(), fake_HDR_Recon.max()], [0, 1]).astype(fake_HDR_Recon.dtype)
            #     # gene_nump = (gene_nump.clip(0, 1))
            #     # gene_nump = tonemapReinhard.process(gene_nump)
            #     # gene_nump = (gene_nump.clip(0, 1) * 255).astype(np.uint8)
            #     ensure_dir(join(outputs_dir.replace('ours/','ours_all/'), 'dynamic', scene_number, str(i).zfill(5)))
            #     cv2.imwrite(join(outputs_dir.replace('ours/','ours_all/'), 'dynamic', scene_number, str(i).zfill(5), str(time_num).zfill(2) + '.hdr'), fake_HDR_Recon)
            # # cv2.imwrite(join(outputs_dir, 'real_world', 'dynamic', scene_number, str(i),'LDRB.png'), LDRB_in)
            #     #  * 255).astype(int)
