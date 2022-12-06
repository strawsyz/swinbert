#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 13:28
# @Author  : strawsyz
# @File    : UCFcrim_dataset.py
# @desc:
import sys

import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
import os
import time
from tqdm import tqdm
from PIL import Image
import cv2
from IPython.display import clear_output, display, HTML

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def dataset_analysis():
    # dataset_root_path = r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-1"
    root_paths = [r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-1",
                  r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-2",
                  r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-3",
                  r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-4"]
    # dataset_root_path = r"/workspace/datasets/ucf-crime/Anomaly-Videos-Part-1/Abuse"
    # print ("num of samples", len(os.listdir(dataset_root_path)))
    for root_path in root_paths:
        for file in os.listdir(root_path):
            class_name = file
            num_samples = len(os.listdir(os.path.join(root_path, file)))
            print(class_name, num_samples)
    #         tmp_filepath =  os.path.join(dataset_root_path, file)
    #         print(tmp_filepath)


def get_annotation():
    annotation_filepath = r"/workspace/datasets/ucf-crime/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
    with open(annotation_filepath) as f:
        content = f.readlines()
    num_frames = 0
    annotations = []
    for sample in content:
        sample = sample.split("  ")
        num_frames += (int(sample[3]) - int(sample[2])) + (int(sample[5]) - int(sample[4]))
        annotations.append((sample[0], sample[1], int(sample[2]), int(sample[3]), int(sample[4]), int(sample[5])))
    return annotations
    # print(num_frames)


def make_directory(dir_path, mode=0o777):
    """create a directory if not exist"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, mode=mode)


def generate_anomaly_sample(sample):
    video_root_path = "/workspace/datasets/ucf-crime/Anomaly-Videos"
    my_ucf_crime_root_path = r"/workspace/datasets/ucf-crime/custom_dataset/"
    filename, class_name, f1, f2, f3, f4 = sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]
    if class_name == "Normal":
        video_path = os.path.join(r"/workspace/datasets/ucf-crime", "Testing_Normal_Videos_Anomaly", filename)
    else:
        return
        video_path = os.path.join(video_root_path, class_name, filename)
    assert os.path.exists(video_path), video_path
    cap = cv2.VideoCapture(video_path)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if class_name == "Normal":
        frameToStart = 0
        frametoStop = total_frame - 1
    else:
        frameToStart = f1
        frametoStop = f2

    currentframe = frameToStart
    num_video_clips = 0
    while currentframe + 64 < frametoStop:
        make_directory(os.path.join(my_ucf_crime_root_path, class_name))
        target_video_filepath = os.path.join(my_ucf_crime_root_path, class_name,
                                             filename[:-4] + "_" + str(num_video_clips) + ".mp4")
        clip_video(video_path, target_video_filepath=target_video_filepath, start_frame=currentframe,
                   end_frame=currentframe + 64)
        currentframe += 64
        num_video_clips += 1

    if f3 != -1:
        frameToStart = f3
        frametoStop = f4

        currentframe = frameToStart
        while currentframe + 64 < frametoStop:
            make_directory(os.path.join(my_ucf_crime_root_path, class_name))
            target_video_filepath = os.path.join(my_ucf_crime_root_path, class_name,
                                                 filename[:-4] + "_" + str(num_video_clips) + ".mp4")
            clip_video(video_path, target_video_filepath=target_video_filepath, start_frame=currentframe,
                       end_frame=currentframe + 64)
            currentframe += 64
            num_video_clips += 1


def clip_video(video_path, target_video_filepath, start_frame, end_frame):
    if os.path.exists(target_video_filepath):
        print("{} exists".format(target_video_filepath))
        return
    cap = cv2.VideoCapture(video_path)

    FPS = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter(target_video_filepath, cv2.VideoWriter_fourcc(*'mp4v'), FPS, size)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)  # 设置读取的位置,从第几帧开始读取视频
    COUNT = 0
    while True:
        success, frame = cap.read()
        if success:
            COUNT += 1
            if COUNT <= end_frame and COUNT > start_frame:  # 选取起始帧
                # print('correct= ', COUNT)
                videoWriter.write(frame)
        else:
            print("Cap read result", success)
            print("total_frame", total_frame)
            print("COUNT", COUNT)
            sys.exit()
        # print('mistake= ', COUNT)
        if COUNT > end_frame:
            break
    print(f'{target_video_filepath} done')


def split_dataset():
    root_path = r"/workspace/datasets/ucf-crime/custom_dataset/"
    filepaths = []
    for class_name in os.listdir(root_path):
        class_filepath = os.path.join(root_path, class_name)
        if os.path.isdir(class_filepath):
            for filename in os.listdir(class_filepath):
                filepaths.append(os.path.join(class_name, filename))
    np.random.shuffle(filepaths)
    num_train = int(0.6 * len(filepaths))
    num_eval = int(0.8 * len(filepaths)) - num_train
    num_test = int(len(filepaths)) - num_eval
    train_data = filepaths[:num_train]
    eval_data = filepaths[num_train:num_train + num_eval]
    test_data = filepaths[num_train + num_eval:]
    # print(train_data)
    # print(eval_data)
    # print(test_data)
    data_split = {"train": train_data, "eval": eval_data, "test": test_data}
    np.save(os.path.join(root_path, "data_split"), data_split)


def custom_ucf_crime_dataset():
    npy_filepath = r"/workspace/datasets/ucf-crime/custom_dataset/data_split.npy"
    annotation = np.load(npy_filepath, allow_pickle=True).tolist()
    # print(annotation)
    filepaths = []
    filepaths.extend(annotation['train'])
    filepaths.extend(annotation['eval'])
    filepaths.extend(annotation['test'])
    return filepaths


def check_data():
    annotation_path = r"/workspace/datasets/ucf-crime/train.npy"
    data = np.load(annotation_path, allow_pickle=True).tolist()
    print(data[0])


def generate_filelist_from_annoataion():
    annotations = get_annotation()
    print(annotations)
    for sample in annotations:
        generate_anomaly_sample(sample)


def analysis_custom_dataset():
    root_path = r"/workspace/datasets/ucf-crime/test.npy"
    # root_path = r"/workspace/datasets/ucf-crime/eval.npy"
    # file = os.listdir(root_path)[0]
    train = np.load(root_path, allow_pickle=True)

    sample = train[0]
    class_names = []
    for sample in train:
        video_path = sample[0]
        print(video_path)
        class_name = video_path.split("/")[-2]
        caption = sample[1][0][0]
        confidence = sample[1][0][1]
        feature = sample[1][0][2]
        class_names.append(class_name)
        # print(class_name)
        # print(sample)
        # print(caption)
        # print(confidence)
        # print(feature)
    from collections import Counter
    print(Counter(class_names))


def get_class_name_from_filepath(filepath):
    return filepath.split("/")[0]


def analysis_custom_dataset2():
    root_path = r"/workspace/datasets/ucf-crime/custom_dataset/data_split.npy"
    datasets = np.load(root_path, allow_pickle=True).tolist()
    train = datasets['train']
    test = datasets['test']
    eval = datasets['eval']
    print(train[0])
    class_names = []
    for sample in train:
        class_name = get_class_name_from_filepath(sample)
        class_names.append(class_name)
    from collections import Counter
    print(Counter(class_names))


def plot_bar_from_dict(dict_):
    keys, values = [], []
    plt.figure(figsize=(15, 6), dpi=160)
    for key, value in dict_.items():
        keys.append(key)
        values.append(value)
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), keys)
    plt.show()


def analysis_custom_dataset3():
    from collections import Counter
    classes_train = []
    classes_test = []
    classes_eval = []
    train_path = r"/workspace/datasets/ucf-crime/train.npy"
    test_path = r"/workspace/datasets/ucf-crime/test.npy"
    eval_path = r"/workspace/datasets/ucf-crime/eval.npy"
    train_samples = np.load(train_path, allow_pickle=True)
    test_samples = np.load(test_path, allow_pickle=True)
    eval_samples = np.load(eval_path, allow_pickle=True)

    def get_class_name(sample):
        return sample[0].split("/")[-2]

    for train_sample in train_samples:
        classes_train.append(get_class_name(train_sample))
    counter_train = Counter(classes_train)
    for eval_sample in eval_samples:
        classes_eval.append(get_class_name(eval_sample))
    counter_eval = Counter(classes_eval)
    for test_sample in test_samples:
        classes_test.append(get_class_name(test_sample))
    counter_test = Counter(classes_test)
    counter = counter_train + counter_eval + counter_test
    plot_bar_from_dict(dict(counter))


def test_annotation():
    """检查添加标注是否一致
    结果，动画内容都一致
    """
    # 检查异常动画的位置是否一致
    annotation_file = r"/workspace/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/test_anomalyv2.txt"
    data_list = []
    root_path = r"/workspace/datasets/ucf-crime/Anomaly-Videos"

    with open(annotation_file) as f:
        data_list = f.readlines()

    for sample in data_list:
        filepath = sample.split("|")[0]
        filepath = os.path.join(root_path, filepath)
        if not os.path.exists(filepath):
            print(filepath)

    # 检查正常动画的位置是否一致
    annotation_file = r"/workspace/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/test_normalv2.txt"
    data_list = []
    root_path = r"/workspace/datasets/ucf-crime"

    with open(annotation_file) as f:
        data_list = f.readlines()

    for sample in data_list:
        filepath = sample.split(" ")[0]
        filepath = os.path.join(root_path, filepath)
        filepath = filepath.replace("Normal_Videos_event", "Testing_Normal_Videos_Anomaly")
        if not os.path.exists(filepath):
            print(filepath)


def split_integer(m, n):
    assert n > 0
    quotient = int(m / n)
    remainder = m % n
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


def generate_samples_4_compare(root_path=None, target_path=None, filelist=None):
    """生成数据集合其他数据进行比较"""
    if root_path is None:
        root_path = r""
    if target_path is None:
        target_path = r""
    if filelist is None:
        filelist = os.listdir(root_path)

    for file in filelist:
        filepath = os.path.join(root_path, file)
        target_filepath = os.path.join(target_path, file)
        if os.path.isdir(filepath):
            make_directory(target_filepath)
            generate_samples_4_compare(filepath, target_filepath)
        else:
            # generate sample in target_path
            cap = cv2.VideoCapture(filepath)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indexes = split_integer(total_frame, 32)
            start = 0
            for ind_, index in enumerate(indexes):
                assert ".mp4" in filepath
                target_video_filepath = target_filepath.replace(".mp4", "-{}.mp4".format(ind_))
                make_directory(os.path.dirname(target_video_filepath))
                clip_video(filepath, target_video_filepath, start, start + index - 1)
                start += index


def check_captions():
    file_path = r"/workspace/datasets/ucf-crime/custom_anno_2/Fighting.npy"
    captions = np.load(file_path, allow_pickle=True)
    for sample in captions:
        print(sample[1][0][0])


def plot_gt(frames, total_frame):
    # total_frame = 200
    # frames = [0, 10, 100, 110]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(frames) // 2):
        ax.add_patch(
            patches.Rectangle(
                (frames[i * 2], 0),
                frames[i * 2 + 1] - frames[i * 2],
                1,
            )
        )
    plt.xlim(0, total_frame)
    plt.ylim((0, 1))

    fig.show()


def generate_annotation_plot(sample_id):
    """generation gt plot according to sample id"""
    annotation_path = r"/workspace/datasets/ucf-crime/test_anomalyv2.txt"
    video_root_path = r"/workspace/datasets/ucf-crime/custom_dataset_2"
    caption_annotation_path = r"/workspace/datasets/ucf-crime/custom_anno_2/{}.npy"
    with open(annotation_path) as f:
        samples = f.readlines()
    sample = samples[sample_id]
    name, total_frame, frames = sample.split('|')[0], int(sample.split('|')[1]), \
                                sample.split('|')[2][1:-2].split(',')
    frames = [int(i) for i in frames]
    print(name, total_frame, frames)
    # plot_gt(frames, total_frame)
    video_path = os.path.join(video_root_path, name)
    class_name = name.split("/")[0]
    filename = name.split("/")[-1].replace(".mp4", "")
    caption_annotation_path = caption_annotation_path.format(class_name)
    all_captions = np.load(caption_annotation_path, allow_pickle=True)
    snip_paths = []
    captions = []
    for sample in all_captions:
        if filename in sample[0]:
            captions.append(sample[1][0][0])
            snip_paths.append(sample[0])
    print(captions)
    print(snip_paths)


if __name__ == '__main__':
    # check_captions()
    generate_annotation_plot(sample_id=0)

    # test_annotation()
    #  generate 32 clipe for each normal video
    # root_path = "/workspace/datasets/ucf-crime/Testing_Normal_Videos_Anomaly"
    # target_path = r"/workspace/datasets/ucf-crime/custom_dataset_2"
    # generate_samples_4_compare(root_path, target_path)

    #     generate 32 clips for each anomaly video
    # root_path = "/workspace/datasets/ucf-crime"
    # target_path = r"/workspace/datasets/ucf-crime/custom_dataset_2"
    #
    # annotation_file = r"/workspace/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/UCF-Crime/test_anomalyv2.txt"
    # data_list = []
    # file_list = []
    # root_path = r"/workspace/datasets/ucf-crime/Anomaly-Videos"
    # with open(annotation_file) as f:
    #     data_list = f.readlines()
    # for sample in data_list:
    #     filepath = sample.split("|")[0]
    #     file_list.append(filepath)
    #
    # generate_samples_4_compare(root_path, target_path, filelist=file_list)
    # video_path = r"/workspace/datasets/ucf-crime/Anomaly-Videos/Abuse/Abuse028_x264.mp4"
    # video_savepath = r"/workspace/tmp/temp.mp4"
    # generate_filelist_from_annoataion()

    # analysis_custom_dataset3()

    # split_dataset()

    # custom_ucf_crime_dataset()
    # train_path = r"C:\Users\syz11\Downloads\train.npy"
    # eval_path = r"C:\Users\syz11\Downloads\eval.npy"
    # test_path = r"C:\Users\syz11\Downloads\test.npy"
    # anomaly_class_name = ["Abuse", "Arrest", "Arson", "Assault", "Burglary", "Explosion", "Fighting", "RoadAccidents", "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalis"]
