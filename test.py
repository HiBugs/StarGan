#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/1/7
@Description:
"""
import os
import cv2
import copy
import numpy as np
import torch
import torch.nn.functional as F
from model import Generator, Discriminator
from torchvision.utils import save_image
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def label2onehot(labels, dim):
    """Convert label indices to one-hot vectors."""
    batch_size = labels.size(0)
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels.long()] = 1
    return out


def create_labels(batch_size, c_dim=5):
    """Generate target domain labels for debugging and testing."""
    c_trg_list = []
    for i in range(c_dim):
        c_trg = label2onehot(torch.ones(batch_size)*i, c_dim)
        c_trg_list.append(c_trg.to(device))
    return c_trg_list


def denorm(x, is_uint=False):
    """Convert the range from [-1, 1] to [0, 1]."""
    out = (x + 1) / 2
    out.clamp_(0, 1)
    if is_uint:
        out *= 255
    return out


g_conv_dim = 64
c_dim = 7
g_repeat_num = 6
g_path = os.path.join('stargan/models', '180000-G.ckpt')
g_model = Generator(g_conv_dim, c_dim, g_repeat_num)
g_model.to(device)
g_model.load_state_dict(torch.load(g_path, map_location=lambda storage, loc: storage))

img_path = "images/test_0079_aligned.jpg"
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (100, 100))

x_real = (image - 127.5) / 127.5

with torch.no_grad():
    x_real = torch.tensor(np.expand_dims(np.transpose(x_real, (2, 0, 1)), axis=0), dtype=torch.float32)
    x_real = x_real.to(device)

    c_trg_list = create_labels(1, c_dim)
    x_fake_list = []
    for n, c_trg in enumerate(c_trg_list):
        c_trg = c_trg.view(c_trg.size(0), c_trg.size(1), 1, 1)
        c = c_trg.repeat(1, 1, x_real.size(2), x_real.size(3))
        x = torch.cat([x_real, c], dim=1)
        x_fake = g_model(x_real, c_trg)

        x_fake_list.append(x_fake)

    x_concat = torch.cat(x_fake_list, dim=3)
    result_path = os.path.join('stargan/results', 'output.jpg')
    save_image(denorm(x_concat.data.cpu()), result_path, nrow=1, padding=0)
    print('Saved real and fake images into {}...'.format(result_path))
