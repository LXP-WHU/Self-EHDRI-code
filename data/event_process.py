#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ZhangX
"""
# import util
import argparse
import glob as gb
import os
import numpy as np
# from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def events_to_PN(events, width, height):
    img_size = (height, width)
    y = events[:, 2].astype(int)
    x = events[:, 1].astype(int)
    pol = events[:, 3]
    img_pos = np.zeros(img_size)
    img_neg = np.zeros(img_size)
    for i in range(events.shape[0]):
        if (pol[i] > 0):
            img_pos[y[i], x[i]] += 1  # count events
        else:
            img_neg[y[i], x[i]] += 1
    img_pos = img_pos.astype(float)
    img_neg = img_neg.astype(float)
    img_pos = np.expand_dims(img_pos, axis=0)
    img_neg = np.expand_dims(img_neg, axis=0)
    event_out = np.concatenate([img_pos, img_neg], axis=0)
    return event_out

def events_to_timesurface(events_in, width, height, endtime):
    events = events_in
    img_size = (height, width)
    sae_pos = np.zeros(img_size, np.float32)
    sae_neg = np.zeros(img_size, np.float32)
    if not events == []:
        timestamp = events[:, 0].astype(np.float32)
        y = events[:, 2].astype(int)
        x = events[:, 1].astype(int)
        pol = events[:, 3]
        t_ref = endtime
        tau = 2000  # decay parameter (in seconds)
        for i in range(events.shape[0]):
            if (pol[i] > 0):
                sae_pos[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)
            else:
                sae_neg[y[i], x[i]] = np.exp(-(t_ref - timestamp[i]) / tau)
    sae_pos = np.expand_dims(sae_pos, axis=0)
    sae_neg = np.expand_dims(sae_neg, axis=0)
    event_out = np.concatenate([sae_pos, sae_neg], axis=0)
    return event_out

def fold_time_dim(inp):
    if inp.ndim == 4:
        T, C, H, W = inp.shape
        out = inp.reshape((T * C, H, W))  # [T,C,H,W] -> [T*C,H,W]
    elif inp.ndim == 5:
        N, T, C, H, W = inp.shape
        out = inp.reshape((N, T * C, H, W))  # [N,T,C,H,W] -> [N,T*C,H,W]
    return out

def event_single_intergral_flow(event_in, span):
    ## generate event frames for sharp-event loss
    event = event_in
    start, end = span
    x, y, p, t = filter_events(event, start, end)  # filter events by temporal dim
    event_window = np.zeros((len(t), 4))
    event_window[:, 0] = t
    event_window[:, 1] = x
    event_window[:, 2] = y
    event_window[:, 3] = p
    events_PN = events_to_PN(np.copy(event_window), width=256, height=256)
    events_timesurface = events_to_timesurface(np.copy(event_window), width=256, height=256, endtime=end)
    event_flow = np.concatenate((events_PN, events_timesurface), axis=0)
    return event_flow

def event_single_intergral(event_in, img_size, span):
    event = event_in
    ## generate event frames for sharp-event loss
    start, end = span
    H, W = img_size
    event_img = np.zeros((H, W)).ravel()
    x, y, p, t = filter_events(event, start, end)  # filter events by temporal dim
    np.add.at(event_img, x + y * W, p)
    event_img = event_img.reshape((H, W))
    return event_img

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

def event_load(temp_event):
    num_frames =26
    temp_event_data = {'t': temp_event[:, 0], 'x': temp_event[:, 1], 'y': temp_event[:, 2],
                       'p': temp_event[:, 3]}
    temp_event_data['p'][temp_event_data['p'] == 0] = -1  # neg in channel 1

    exp_start_leftB = temp_event_data['t'][0]
    exp_end_rightB = temp_event_data['t'][-1]
    span_B = (exp_start_leftB, exp_end_rightB)
    img_size = (256, 256)
    ## generate target timestamps
    timestamps = np.linspace(exp_start_leftB, exp_end_rightB, num_frames,
                             endpoint=True)  # include the last frame
    ## initialize lists
    leftB_inp = []
    rightB_inp = []

    for j in range(len(timestamps)):
        ts = timestamps[j]
        ## for left blurry image
        leftB_tmp, rightB_tmp = event2frame(temp_event_data, img_size, ts, span_B, 8)

        leftB_tmp = fold_time_dim(leftB_tmp)
        rightB_tmp = fold_time_dim(rightB_tmp)
        leftB_inp.append(leftB_tmp)
        rightB_inp.append(rightB_tmp)
    leftB_inp = np.array(leftB_inp)
    rightB_inp = np.array(rightB_inp)
    return leftB_inp, rightB_inp