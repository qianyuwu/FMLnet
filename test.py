# Copyright (c) 2020 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file contains content licensed by https://github.com/xinntao/BasicSR/blob/master/LICENSE/LICENSE


import glob
import sys
from collections import OrderedDict

from natsort import natsort

import options.options as option
from Measure import Measure, psnr
from imresize import imresize
from models import create_model
import torch
from utils.util import opt_get
import numpy as np
import pandas as pd
import os
import cv2


def fiFindByWildcard(wildcard):
    return natsort.natsorted(glob.glob(wildcard, recursive=True))


def load_model(conf_path):
    opt = option.parse(conf_path, is_train=False)
    opt['gpu_ids'] = None
    opt = option.dict_to_nonedict(opt)
    model = create_model(opt)

    model_path = opt_get(opt, ['model_path'], None)
    model.load_network(load_path=model_path, network=model.netG)
    return model, opt


def predict(model, lr):
    model.feed_data({"LQ": t(lr)}, need_GT=False)
    model.test()
    visuals = model.get_current_visuals(need_GT=False)
    return visuals.get('rlt', visuals.get("SR"))


def t(array): return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t): return (
        np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(
    np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape

    h_start = max(h // 2 - size // 2, 0)
    h_end = min(h_start + size, h)

    w_start = max(w // 2 - size // 2, 0)
    w_end = min(w_start + size, w)

    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], 'reflect')


def main():
    conf_path = sys.argv[1]
    conf = conf_path.split('/')[-1].replace('.yml', '')
    model, opt = load_model(conf_path)

    lr_dir = opt['dataroot_LR']
  #  hr_dir = opt['dataroot_GT']

    lr_paths = fiFindByWildcard(os.path.join(lr_dir, '*.png'))
   # hr_paths = fiFindByWildcard(os.path.join(hr_dir, '*.png'))

    this_dir = os.path.dirname(os.path.realpath(__file__))
    test_dir = os.path.join(this_dir, '..', 'results80000_testx4', conf)

    scale = opt['scale']

    pad_factor = 8



    for lr_path, idx_test in zip(lr_paths, range(len(lr_paths))):

        lr = imread(lr_path)
       # hr = imread(hr_path)

        # Pad image to be % 2
        h, w, c = lr.shape
        print('*********************************lr*********************')
        print(lr.shape)

        lq_orig = lr.copy()
        lr = impad(lr, bottom=int(np.ceil(h / pad_factor) * pad_factor - h),
                   right=int(np.ceil(w / pad_factor) * pad_factor - w))

        lr_t = t(lr)
        print('********************************lr_t*********************')
        print(lr_t.shape)
        heat = opt['heat']
        n_outputs= opt['n_outputs']        
  

        for s in range(n_outputs):   

            sr_t = model.get_sr(lq=lr_t, heat=heat)

            sr = rgb(torch.clamp(sr_t, 0, 1))
            sr = sr[:h * scale, :w * scale]

            path_out_sr = os.path.join(test_dir, "{:0.2f}".format(heat).replace('.', ''), "{:06d}_sample{:05d}.png".format(idx_test,s))
            imwrite(path_out_sr, sr)

 


if __name__ == "__main__":
    main()
