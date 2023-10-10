# encoding:utf-8
"""
去除轨迹点小于15的轨迹段
进行列表随机 按照一定规则进行数据采样 预计最后数据量在500m左右 取样百分之十
更新最大最小lon lat cog sog dis的列表
对这五个字段进行归一化操作
按照6:2:2划分为train,val,test集
"""

# 存放原始数据文件的目录
import json
import os
import random
import numpy as np

from tqdm import tqdm


def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic


def get_list(dire=None):
    ten_percent_list = []  # 总的航行段数据列表
    a=dire
    for root, dirs, files in os.walk(dire):
        print('start')
        for file in tqdm(files):
            path = os.path.join(root, file)
            file_name = file.replace('.json', '')  # 文件名，如：east_1
            with open(path, 'r') as f:
                load_dict = json.load(f)

            ship_data = load_dict.get("ship_data")  # 船舶航行段数据
            # 按照随机数划分百分之十数据到列表中
            for a in ship_data.values():
                for b in a:
                    if len(b) > 15:
                        ten_percent_list.append(b)
    return ten_percent_list


def normalization(data=None, min_=None, max_=None):
    data = float(data)
    new_a = (data - min_) / (max_ - min_)
    return new_a


def normalize(list_data=None):
    # 排序为 lat,lon,cog,sog,dis
    max_list = [-11000, -10000, -10000, -10000, -10000]
    min_list = [11000, 10000, 10000, 10000, 10000]
    for run_period in list_data:
        for gjd in run_period:
            currt_lat = float(gjd[1])
            if currt_lat > max_list[0]:
                max_list[0] = currt_lat
            elif currt_lat < min_list[0]:
                min_list[0] = currt_lat

            currt_lon = float(gjd[2])
            if currt_lon > max_list[1]:
                max_list[1] = currt_lon
            elif currt_lon < min_list[1]:
                min_list[1] = currt_lon

            currt_cog = float(gjd[3])
            if currt_cog > max_list[2]:
                max_list[2] = currt_cog
            elif currt_cog < min_list[2]:
                min_list[2] = currt_cog

            currt_sog = float(gjd[4])
            if currt_sog > max_list[3]:
                max_list[3] = currt_sog
            elif currt_sog < min_list[3]:
                min_list[3] = currt_sog

            currt_dis = float(gjd[5])
            if currt_dis > max_list[4]:
                max_list[4] = currt_dis
            elif currt_dis < min_list[4]:
                min_list[4] = currt_dis

    for running_period in list_data:
        for gjd in running_period:
            nor_lat = normalization(gjd[1], min_list[0], max_list[0])
            nor_lon = normalization(gjd[2], min_list[1], max_list[1])
            nor_cog = normalization(gjd[3], min_list[2], max_list[2])
            nor_sog = normalization(gjd[4], min_list[3], max_list[3])
            nor_dis = normalization(gjd[5], min_list[4], max_list[4])
            gjd[1] = nor_lat
            gjd[2] = nor_lon
            gjd[3] = nor_cog
            gjd[4] = nor_sog
            gjd[5] = nor_dis
    return list_data, [max_list, min_list]


def gen_dataset():
    directory = r'../dataset'
    save_dire = r'../dataset_json/'
    #get_list():筛选出每段大于15个轨迹点的轨迹段
    ship_running_list = get_list(directory)
    # 对所有轨迹进行一遍清洗
    # 对所有列表中的数据进行归一化操作
    normalized_list, m_n = normalize(ship_running_list)
    random.shuffle(normalized_list)
    train = normalized_list[:int(len(normalized_list) * 0.8)]
    val = normalized_list[int(len(normalized_list) * 0.8):int(len(normalized_list) * 0.9)]
    test = normalized_list[int(len(normalized_list) * 0.9):int(len(normalized_list))]
    data_list = [train, val, test, m_n]
    for index, file in enumerate(data_list):
        if index == 0:
            file_name = 'train'
        elif index == 1:
            file_name = 'val'
        elif index == 2:
            file_name = 'test'
        elif index == 3:
            file_name = 'max_min'
        else:
            file_name = ''
        with open(save_dire + file_name + '.json', 'w') as f:
            json.dump(file, f)
    return 'success get dataset'



