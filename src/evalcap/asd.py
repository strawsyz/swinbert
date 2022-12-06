# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2022/9/21 13:14
# # @Author  : strawsyz
# # @File    : asd.py
# # @desc:
#
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import numpy as np
from coco_caption.pycocotools.coco import COCO
import json


def create_sample():
    root_path = r"/workspace/datasets/a2d_sentences/test"
    filepaths = os.listdir(root_path)
    filepaths = [os.path.join(root_path, file) for file in filepaths]

    def get_information(path):
        data = np.load(path, allow_pickle=True)
        caption = data[0]
        video_path = data[3][3]
        return caption, video_path

    json_content = {}
    annotations = []
    images = []
    for idx, filepath in enumerate(filepaths):
        caption, video_id = get_information(filepath)
        annotations.append({'image_id': video_id, 'caption': caption, "id": idx})
        images.append({'id': video_id, 'file_name': video_id})
    json_content['annotations'] = annotations
    json_content['images'] = images
    json_content['type'] = 'captions'
    json_content['info'] = 'dummy'
    json_content['licenses'] = 'dummy'
    print(json_content)
    save_path = r"/workspace/SwinBERT/datasets/MSVD/a2d_val.caption_coco_format.json"

    json_content = json.dumps(json_content)
    with open(save_path, 'w') as f:
        f.write(json_content)
        f.close()

# print(label_file_coco)  # datasets/MSVD/val.caption_coco_format.json
label_file_coco = r"/workspace/SwinBERT/datasets/MSVD/val.caption_coco_format.json"
res_file_coco = r"/workspace/SwinBERT/output/checkpoint-1-1/pred.MSVD.val.beam1.max20_coco_format.json"
# print(res_file_coco)  # ./output/checkpoint-1-1/pred.MSVD.val.beam1.max20_coco_format.json
# coco = COCO(label_file_coco)
save_path = r"/workspace/SwinBERT/datasets/MSVD/a2d_val.caption_coco_format.json"

coco = COCO(save_path)


# # cocoRes = coco.loadRes(res_file_coco)
# anns = json.load(open(res_file_coco))
# imageIds = [ann['image_id'] for ann in anns]
# print(imageIds)
# print(anns)


# res = json.load(open(label_file_coco))
# print(len(res['annotations']))
# print(res['annotations'][0])
# print(res['annotations'][1])
# print(res['annotations'][2])
#
# print(len(res['images']))
# print(res['images'][0])
# print(res['images'][1])
# print(res['images'][2])


if __name__ == '__main__':
    create_sample()
# print(res['type'])
# print(res['info'])
# print(res['licenses'])
# for key in res.keys():
#     print(key)
# print(res)

# class PretrainDataset(Dataset):
#     def __init__(self, abnormal_feats, normal_feats):
#         super(PretrainDataset, self).__init__()
#         self.abnormal = np.array(abnormal_feats)
#         if len(self.abnormal.shape) >= 4 or len(self.abnormal.shape) == 1:
#             # num_class x num_video x num_seg x feat_dim --> num_class * num_video x num_seg x feat_dim
#             self.abnormal = np.concatenate(self.abnormal)
#         self.normal = np.array(normal_feats)
#         self.length = min(len(self.normal), len(self.abnormal))
#
#     def shuffle(self):
#         ab_indices = np.random.permutation(len(self.abnormal))
#         no_indices = np.random.permutation(len(self.normal))
#         self.abnormal = self.abnormal[ab_indices]
#         self.normal = self.normal[no_indices]
#
#     def __getitem__(self, item):
#         return np.concatenate([self.abnormal[item], self.normal[item]]), np.array([1, 0])
#
#     def __len__(self):
#         return self.length
#
#
# class EvalDataset(Dataset):
#     def __init__(self, feat_dict, gt_dict, frame_dict):
#         self.feats = []
#         self.gts = []
#         self.frames = []
#         self.names = []
#         for cls, dic in feat_dict.items():
#             if type(dic) == dict:
#                 # for abnormal video
#                 for video_name, feats in dic.items():
#                     gt = gt_dict[video_name]['gt']
#                     if len(gt) == 0:
#                         print(video_name)
#                         continue
#                     self.feats.append(feats)
#                     for i in range(len(gt)):
#                         gt[i] = int(gt[i])
#                     self.gts.append(gt)
#                     self.frames.append(int(frame_dict[video_name]))
#                     self.names.append(video_name)
#             else:
#                 # for normal video
#                 self.feats.append(dic) # 'dic' is normal features
#                 self.frames.append(int(frame_dict[cls])) # 'cls' is frame number
#                 self.gts.append([-1, -1])
#                 self.names.append(cls)
#
#     def __getitem__(self, item):
#         return self.feats[item], self.gts[item], self.frames[item], self.names[item]
#
#     def __len__(self):
#         return len(self.frames)
#
#     def collate_fn(self, data):
#         feats, gts, frames, names = list(zip(*data))
#         feats, gts, frames, names = torch.Tensor(np.asarray(feats)), list(gts), list(frames), list(names)
#         return feats, gts, frames, names
#
#
# class MetaDataset(Dataset):
#     def __init__(self, abnorm_feats, norm_feats, num_task):
#         self.abnorm_feats = np.array(abnorm_feats)
#         if len(self.abnorm_feats.shape) >= 4 or len(self.abnorm_feats.shape) == 1:
#             self.abnorm_feats = np.concatenate(self.abnorm_feats)
#
#         self.norm_feats = np.array(norm_feats)
#         self.length = min(len(self.abnorm_feats), len(self.norm_feats))
#         self.n_spt = 10
#         self.n_qry = 30
#         self.num_task = num_task
#         self.build_episode()
#
#     def build_episode(self):
#         self.ab_spt = []
#         self.no_spt = []
#         self.ab_qry = []
#         self.no_qry = []
#
#         for i in range(self.num_task):
#             ab_indices = np.random.permutation(len(self.abnorm_feats))
#             no_indices = np.random.permutation(len(self.norm_feats))
#
#             self.ab_spt.append(self.abnorm_feats[ab_indices[:self.n_spt]])
#             self.no_spt.append(self.norm_feats[no_indices[:self.n_spt]])
#             self.ab_qry.append(self.abnorm_feats[ab_indices[self.n_spt:self.n_spt+self.n_qry]])
#             self.no_qry.append(self.norm_feats[no_indices[self.n_spt:self.n_spt+self.n_qry]])
#
#     def __getitem__(self, item):
#         return np.concatenate([self.ab_spt[item], self.no_spt[item]]), np.array([1] * self.n_spt + [0] * self.n_spt),\
#                np.concatenate([self.ab_qry[item], self.no_qry[item]]), np.array([1] * self.n_qry + [0] * self.n_qry)
#
#     def __len__(self):
#         return self.num_task
#
#
#
#
# import numpy as np
# import cv2
# import os
# import time
#
# START_HOUR = 0
# START_MIN = 6
# START_SECOND = 55
# START_TIME = START_HOUR * 3600 + START_MIN * 60 + START_SECOND  # 设置开始时间(单位秒)
# END_HOUR = 1
# END_MIN = 6
# END_SECOND = 55
# END_TIME = END_HOUR * 3600 + END_MIN * 60 + END_SECOND  # 设置结束时间(单位秒)
#
#
# video_path = "./20210225_15.mp4"
# cap = cv2.VideoCapture(video_path)
# FPS = cap.get(cv2.CAP_PROP_FPS)
# print(FPS)
# size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# size = (1920,1080)
# print(size)
# TOTAL_FRAME = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频总帧数
# frameToStart = START_TIME * FPS  # 开始帧 = 开始时间*帧率
# print(frameToStart)
# frametoStop = END_TIME * FPS  # 结束帧 = 结束时间*帧率
# print(frametoStop)
# videoWriter =cv2.VideoWriter('./video/video1.avi',cv2.VideoWriter_fourcc('X','V','I','D'),FPS,size)
#
# # cap.set(cv2.CAP_PROP_POS_FRAMES, frameToStart)  # 设置读取的位置,从第几帧开始读取视频
# COUNT = 0
# while True:
#         success, frame = cap.read()
#         if success:
#             COUNT += 1
#             if COUNT <= frametoStop and COUNT > frameToStart:  # 选取起始帧
#                 print('correct= ', COUNT)
#                 videoWriter.write(frame)
#         # print('mistake= ', COUNT)
#         if COUNT > frametoStop:
#             break
# print('end')
