#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2020/1/9
@Description:
"""
import os
import sys
import shutil
from tqdm import tqdm


def get_filenames(path):
    name = os.listdir(path)
    return name


dataset_path = '../Datasets/RAFdb/'
new_dirs = '../Datasets/RAFdb_TSF'

Emotion_Labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
Usage_Dir = ['train', 'test']

for pre_dir in Usage_Dir:
    for emo_dir in Emotion_Labels:
        type_dir = os.path.join(new_dirs, pre_dir, emo_dir)
        # print(type_dir)
        if not os.path.exists(type_dir):
            os.makedirs(type_dir)
            print("Create ", type_dir)

with open(dataset_path + 'label.txt', 'r+', encoding='utf8') as f:
    images_list = f.readlines()

for image in tqdm(images_list):
    image_name, image_label_str = image.split(' ')[0:2]
    image_label_idx = int(str(image_label_str).replace('\n', '')) - 1
    image_label = Emotion_Labels[image_label_idx]
    usage = image_name.split('_')[0]

    org_path = os.path.join(dataset_path, image_name)
    trg_dir = os.path.join(new_dirs, usage, image_label)

    shutil.copy(org_path, trg_dir)


