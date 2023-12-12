# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os.path
import io
import zipfile
import os.path as osp
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import random
from io import BytesIO
import glob
import torch
import json
import pdb
import time


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


class EventHDR_Dataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        print(opt.name)
        self.dataroot = opt.dataroot
        self.event_number = opt.event_number
        self.is_crop = opt.is_crop
        self.is_flip_rotate = opt.is_flip_rotate
        self.crop_sz_H = 128
        self.crop_sz_W = 128

        ####################################################################################################
        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transform_img = transforms.Compose(transform_list)
        ####################################################################################################
        self.imnames = glob.glob(os.path.join(self.dataroot, '*/*'))
        random.shuffle(self.imnames)

    def __getitem__(self, index):
        index = random.randint(0, len(self.imnames) - 1)
        static_pth = self.imnames[index]
        # start = time.time()
        LDRB = torch.from_numpy(np.load(os.path.join(static_pth,'LDRB.npy'))/ 255.0)
        event_leftB = torch.from_numpy(np.load(os.path.join(static_pth,'event_leftB.npy')))
        event_rightB = torch.from_numpy(np.load(os.path.join(static_pth,'event_rightB.npy')))
        B_all_start = torch.from_numpy(np.load(os.path.join(static_pth,'B_all_start.npy')))
        B_all_end = torch.from_numpy(np.load(os.path.join(static_pth,'B_all_end.npy')))
        exposure = torch.from_numpy(np.load(os.path.join(static_pth,'exposure.npy')))
        LDRtif = torch.from_numpy((np.load(os.path.join(static_pth,'LDRtif.npy'))/ 65535.0))
        LDRtif[0,...] = (LDRtif[0,...]/162*255).clip(0,1)
        LDRtif[1,...] = (LDRtif[1,...]/195*255).clip(0,1)
        LDRtif[2,...] = (LDRtif[2,...]/200*255).clip(0,1)        

        if self.is_crop:
            y = np.random.randint(low=1, high=(LDRB.shape[1] - self.crop_sz_H))
            x = np.random.randint(low=1, high=(LDRB.shape[2] - self.crop_sz_W))
            LDRB = LDRB[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
            event_leftB = event_leftB[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
            event_rightB = event_rightB[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
            B_all_start = B_all_start[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
            B_all_end = B_all_end[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
            LDRtif = LDRtif[:, y:y + self.crop_sz_H, x:x + self.crop_sz_W]
        # end = time.time()
        # print(end-start)
        input_dict = {'LDRB': LDRB, 'event_leftB': event_leftB, 'event_rightB': event_rightB,
                      'B_all_start': B_all_start, 'B_all_end': B_all_end, 'exposure': exposure,
                      'LDRtif': LDRtif}
        return input_dict

    def __len__(self):
        # return len(self.imnames)  ## actually, this is useless, since the selected index is just a random number
        return 8000  ## actually, this is useless, since the selected index is just a random number
    def name(self):
        return 'EventHDR_Dataset'
