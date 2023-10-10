# -*- coding: utf-8 -*- 
# @Time : 2022/10/25 9:52 
# @Author : wsj 
# @File : feature_engineering.py 
# @desc:
import json
import os.path
import random
from preprocess.conf import len_train, len_label, BATCH_SIZE
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import torch.nn as nn


#
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    # item 为数据的随机索引，在0---n-1中自动获取，n为数据的数量
    def __getitem__(self, item):
        train_data1 = self.data[item][0:len_train]
        label_data = self.data[item][len_train:len_train + len_label]
        return torch.Tensor(train_data1), torch.Tensor(label_data)
    #   返回数据集大小
    def __len__(self):
        return len(self.data)


def get_dataloader(data, batch_size=None, shuffle=True):
    dataset = MyDataset(data=data)
    # 组合数据集和采样器，并提供可迭代对象给定数据集
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0, drop_last=True)
    return data_loader


class Input_Module(nn.Module):
    """
    Embedding and concatenating the spatiotemporal semantics， trajectory points, and  driving status
    """
    embed_dim = [("month", 13, 2), ("day", 32, 2), ("hour", 25, 2), ("type", 101, 8)]

    def __init__(self):
        super(Input_Module, self).__init__()
        # 对时间和
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            self.add_module(name + '_embed', nn.Embedding(num_embeddings, embedding_dim))  # ！！！！

    def forward(self, train, label):
        train_loc_semantic, train_property_semantic, train_type_semantic, train_time_semantic = Matrix_slice(train)
        label_data = Matrix_slice(label)
        label_loc = label_data[0]

        attr = {
            "month": train_time_semantic[:, :, 0],
            "day": train_time_semantic[:, :, 1],
            "hour": train_time_semantic[:, :, 2],
            "type": train_type_semantic
        }
        time_semantic = []
        for name, num_embeddings, embedding_dim in Input_Module.embed_dim:
            #   gerattr()返回一个对象属性值
            embed = getattr(self, name + '_embed')
            _attr = embed(attr[name])
            time_semantic.append(_attr)
        # time_type_semantic[128,10,14]
        time_type_semantic = torch.cat(time_semantic, dim=2)  # 3. 时间和类型语义拼接

        # #   将船舶类型划分为100个类别，嵌入到8纬向量中
        # type_embed = torch.nn.Embedding(101, 8)
        # #   label_data[2]为船舶类型的语义信息[128,5]  type_semantic [128,5,8]

        # 将位置语义和速度，角度，航行距离语义拼接
        train_input_tensor = torch.cat((train_loc_semantic, train_property_semantic), dim=2)
        # print('train_loc_semantic:',train_loc_semantic.size())
        # print('train_property_semantic:',train_property_semantic.size())
        # print('label_loc:', label_loc.size())
        # print('type_semantic:', type_semantic.size())
        return train_input_tensor, label_loc, time_type_semantic


def Matrix_slice(data=None):
    """传入三维tensor进行切片操作"""
    #   lat lon
    loc_semantic = data[:, :, 1:3]
    #   角度 速度 航行距离
    property_semantic = data[:, :, 3:6]

    #   船类型
    type_semantic = data[:, :, 6].long()
    #   月 日 小时
    time_semantic = data[:, :, 7:10].long()
    return loc_semantic, property_semantic, type_semantic, time_semantic


def trajectory_cut(data=None):
    """滑动窗口切分轨迹"""
    # 长度15轨迹列表
    list_15 = []

    for running_period in data:
        left = 0
        right = 15
        while right <= len(running_period):
            train_seq = running_period[left:right]
            left += 1
            right += 1
            list_15.append(train_seq)
    return list_15


def gen_dataloader(args):
    trian_path = r'../dataset_server_demo/train.json'
    val_path = r'../dataset_server_demo/val.json'
    test_path = r'../dataset_server_demo/test.json'
    with open(trian_path, 'r') as f:
        train = json.load(f)
    with open(val_path, 'r') as v:
        val = json.load(v)
    with open(test_path, 'r') as u:
        test = json.load(u)

    # 滑动窗进行轨迹切分
    train = trajectory_cut(train)  # 0.8
    val = trajectory_cut(val)  # 0.1
    test = trajectory_cut(test)  # 0.1

    max_min_path = r'../dataset_server_demo/max_min.json'
    with open(max_min_path, 'r') as m:
        max_min = json.load(m)
    max_ = max_min[0]
    min_ = max_min[1]

    train_data = get_dataloader(train, BATCH_SIZE)
    val_data = get_dataloader(val, BATCH_SIZE, False)
    test_data = get_dataloader(test, BATCH_SIZE, False)

    return train_data, val_data, test_data, max_, min_

