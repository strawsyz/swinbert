#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 15:48
# @Author  : strawsyz
# @File    : temp.py
# @desc:
import os

import h5py
import numpy as np


def get_information(path):
    data = np.load(path, allow_pickle=True)
    caption = data[0]
    image = data[1]
    mask = data[2]
    # print(data[3])
    video_id = data[3][0]
    frame_id = data[3][1]
    instance_id = data[3][2]
    filepath = rf"/workspace/datasets/a2d_sentences/text_annotations/a2d_annotation_with_instances/{video_id}/{frame_id:05d}.h5"
    assert os.path.exists(filepath), print(filepath)
    f = h5py.File(filepath)
    instances = list(f['instance'])
    mask = mask[instances.index(instance_id)]
    return caption, image, mask

def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n

if __name__ == '__main__':
    # print(split_integer(1000,32))

    print()
    # import matplotlib.pyplot as plt
    #
    # data = [5, 20, 15, 25, 10]
    #
    # plt.bar(range(len(data)), data)
    # plt.xticks(range(len(data)),
    #            [r'$really\ bad$', r'$bad$', r'$normal$', r'$good$', r'$readly\ good$'])
    # plt.show()

    # path = r"/workspace/datasets/a2d_sentences/Train/a2d_1733.npy"
    # caption, image, mask = get_information(path)
    # print(image.shape)
    # print(mask.shape)
    # print(caption)
    # for file in os.listdirroot_path:

    # root_path = "/workspace/datasets/a2d_sentences/Train"
    # file_list = os.listdir(root_path)
    # print(file_list)
