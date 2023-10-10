# -*- coding: utf-8 -*- 
# @Time : 2022/10/17 16:59 
# @Author : wsj 
# @File : traj_preprocessing.py 
# @desc:
import csv
import math
import time
import os
import json

from scipy import interpolate
# from scipy.misc import derivative
import pandas as pd
import numpy as np

# from conf import *
from tqdm import tqdm

from preprocess.conf import PATH_AIS_TXT, SOG_threshold, divided_TIME, TIMESTAMP_interval, TIMESTAMP_difference, \
    stop_DISTANCE, parking_number, EARTH_RADIUS
from preprocess.get_dataset import gen_dataset


def preprocessing(res=None):
    """
    1. 根据MMSI进行数据分类，排序
    2. 对每个MMSI船舶的轨迹点列表进行清洗，删除重复时间戳的数据，删除跳点数据
    3. 按照时间间隔进行第一次分割
    4. 对分割后的轨迹段列表进行等时间间隔插值处理
    5. 对插值后的轨迹段进行第二次航行段切割

    output: 输出MMSI对应多个航行段的数据
    :param path:
    :return: 返回每艘船分类好的按时间戳排序的数据
    """
    # PATH = '../' + path
    # # 读取txt文件
    # with open(PATH, 'r') as f:
    #     datas = f.readlines()

    # 将分类后的数据放到字典
    ref_classify_dict = res

    # # 拿到除去表头的每一行数据
    # for data in datas[1:]:
    #     data = data.replace('\n', '')  # 去掉每一行的换行符
    #     data = data.split(',')  # 将每一行按逗号组装成列表
    #     MMSI = data[0]  # 根据第一列MMSI值进行分类
    #     # 判断MMSI是否放进字典
    #     if MMSI in ref_classify_dict.keys():
    #         ref_classify_dict[MMSI].append(data)
    #     else:
    #         ref_classify_dict[MMSI] = [data]


    # 按照时间戳进行排序
    for mmsi in tqdm(ref_classify_dict.keys()):
        # 获取船舶的静态特征
        Length, Width, Draft, vessel_type = None, None, None, None
        for i in range(len(ref_classify_dict[mmsi])):
            # Length = ref_classify_dict[mmsi][i][8]
            # Width = ref_classify_dict[mmsi][i][9]
            # Draft = ref_classify_dict[mmsi][i][10]
            vessel_type = ref_classify_dict[mmsi][i][7]
            # if Length and Width and Draft:
            if vessel_type:
                break
        # if not Length:
        #     Length = '船长度未知'
        # if not Width:
        #     Width = '船宽度未知'
        # if not Draft:
        #     Draft = '船吃水深度未知'
        if not vessel_type:
            vessel_type = 100  # 船舶类型为空时，定义编号为 100

        sorted_list = sorted(ref_classify_dict[mmsi], key=lambda x: x[1])  # 按照时间戳进行排序,时间戳位于字段位置的第二个
        timed_lists = clear_list(sorted_list)  # 按照时间间隔阈值分段轨迹段 get: [[1],[2],[3]]
        running_period = []
        # 对按照时间分割的轨迹段进行分段插值
        for timed_list in timed_lists:
            interpolated_list = list_interpolation(timed_list)
            # 进行轨迹分割，按照相邻轨迹点的距离小于阈值(暂定10m)进入停驻段，第一个距离大于阈值跳出停驻段
            segmented_list = list_segmentation(interpolated_list)
            for a in segmented_list:
                running_period.append(a)
        # 根据经纬度计算航速航向，补充到轨迹点字段信息中
        new_running_periods = []
        for running_period_ in running_period:  # 拿到每个航行段，计算每个gjd的速度航向
            # 航行段列表
            current_running_period = []
            # 当前时间戳对应的时间列表
            current_time_list = []
            # 累计航行距离
            all_running_distance = 0
            # 当前时间戳累计航行距离列表
            current_distance_list = [0]
            current_course = None  # 默认航向
            if len(running_period_) >= 2:  # 轨迹点为1，不能计算航向航速
                starting_point_lon_lat = [running_period_[0][2], running_period_[0][1]]
                for current_index, current_gjd in enumerate(running_period_):
                    # 先计算对应轨迹点已经行驶的距离
                    first_time = running_period_[0][0]
                    current_timestamp = current_gjd[0]  # 当前轨迹点的时间戳
                    current_time_list.append((int(current_timestamp) - int(first_time)) / 1000)  # 转换成s
                    current_lon = current_gjd[2]  # 当前轨迹点的经度
                    current_lat = current_gjd[1]  # 当前轨迹点的纬度
                    current_gjd_list = [current_timestamp, current_lat, current_lon]  # 增加字段后的轨迹点
                    next_index = current_index + 1  # 下一轨迹点的索引
                    if next_index < len(running_period_):
                        next_lon = running_period_[next_index][2]  # 下一轨迹点的经度
                        next_lat = running_period_[next_index][1]  # 下一轨迹点的纬度
                        # 算前后两点的距离  km
                        distance = get_distance(current_lon, current_lat, next_lon, next_lat)
                        all_running_distance += distance  # 加和之前的航行距离
                        current_distance_list.append(all_running_distance)  # 对应时间戳添加距离
                        current_course = get_course(current_lon, current_lat, next_lon, next_lat)  # 得到当前点和后一点的航向
                        current_gjd_list.append(current_course)
                    elif next_index == len(running_period_):  # 如果当前点是最后一个航行轨迹点
                        if current_course:
                            last_course = current_course
                        else:
                            last_course = 0
                        current_gjd_list.append(last_course)
                    #原来(时间戳,纬度,经度)-----现在current_running_period为(时间戳,纬度,经度,角度)
                    current_running_period.append(current_gjd_list)
                # print('-----------------------')
                time_variable = pd.Series(current_time_list)
                distance_variable = pd.Series(current_distance_list)
                gx = interpolate.interp1d(time_variable, distance_variable,
                                          fill_value="extrapolate")  # 插值一个一维函数 返回interp1d
                # 注*： fill_value="extrapolate" 外推到范围外，可能会有误差
                #位移对时间求导代表速度，填入对应时间值即可获得对应的速度
                gd = get_derivative(gx)  # 对应位移时间函数的导函数，填入对应的时间值
                # 对应时间戳的速度列表
                v_list = []
                for time_point in current_time_list:
                    current_v = gd(time_point)
                    v_list.append(current_v * 1000)  # 单位km/s 转换成m/s
                new_gjd_list = []
                for index, gjd in enumerate(current_running_period):
                    gjd.append(round(v_list[index], 3))
                    gjd.append(round(current_distance_list[index], 5))
                    timestamp = int(current_running_period[0][0])  # 13位时间戳，转换成月日时的形式
                    data_time = time.localtime(timestamp / 1000)
                    # 轨迹点中加入类型信息
                    gjd.append(int(vessel_type))
                    # 根据日期拿到月日时
                    time_month = data_time.tm_mon
                    time_day = data_time.tm_mday
                    time_hour = data_time.tm_hour
                    gjd.append(time_month)
                    gjd.append(time_day)
                    gjd.append(time_hour)
                    # gjd.append(starting_point_lon_lat)
                    # 轨迹点中加入开始的轨迹点的经纬度
                    gjd.append(running_period_[0][2])
                    gjd.append(running_period_[0][1])
                    # 轨迹点中加入mmsi编号
                    gjd.append(int(mmsi))
                    new_gjd_list.append(gjd)

                new_running_periods.append(new_gjd_list)
        ref_classify_dict[mmsi] = new_running_periods

    # 根据前后两点的速度之差去除跳点：默认开始点为正常点，计算前一点与后一点的速度差值，超过阈值舍去后一点，依次循环计算
    ref_classify_dict = clear_jump_point_by_D_value(ref_classify_dict, D_value=10)

    # 获取经纬度最大值，最小值
    max_list = [-10000, -10000, -10000, -10000, -10000]  # lat,lon
    min_list = [100000, 110000, 110000, 110000, 110000]  # lat,lon
    for mmsi, running_periods in tqdm(ref_classify_dict.items()):
        # if running_periods:
        #     for run_period in running_periods:
        #         if len(run_period) < 15:
        #             running_periods.remove(run_period)
        # if not running_periods:
        #     del ref_classify_dict[mmsi]
        for running_period in running_periods:
            for gjd in running_period:
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
    # 进行归一化
    # for mmsi, running_periods in tqdm(ref_classify_dict.items()):
    #     if running_periods:
    #         for run_period in running_periods:
    #             if len(run_period) < 15:
    #                 running_periods.remove(run_period)
    #     if not running_periods:
    #         del ref_classify_dict[mmsi]
    #     for running_period in running_periods:
    #         for gjd in running_period:
    #             currt_lat = float(gjd[1])
    #             currt_lon = float(gjd[2])
    #             currt_cog = float(gjd[3])
    #             currt_sog = float(gjd[4])
    #             currt_dis = float(gjd[5])
    #             normalized_lat = normalization(currt_lat, min_list[0], max_list[0])
    #             gjd[1] = normalized_lat
    #             normalized_lon = normalization(currt_lon, min_list[1], max_list[1])
    #             gjd[2] = normalized_lon
    #             normalized_cog = normalization(currt_cog, min_list[2], max_list[2])
    #             gjd[3] = normalized_cog
    #             normalized_sog = normalization(currt_sog, min_list[3], max_list[3])
    #             gjd[4] = normalized_sog
    #             normalized_dis = normalization(currt_dis, min_list[4], max_list[4])
    #             gjd[5] = normalized_dis
    # # print('----------------------')
    res = {
        "ship_data": ref_classify_dict,
        "max_min": [max_list, min_list]
    }

    return res


def clear_list(gjd_list=None, max_speed=SOG_threshold, time_interval=divided_TIME):
    """
    1. 删除时间戳重复的数据，保留时间戳第一次出现的数据
    2. 跳点处理：
        (1) 在下一轨迹点与当前轨迹点的平均速度超过阈值时，按跳点处理，默认删除
        (2) 相邻轨迹点时间差大于阈值(6h)，分割列表

    :param time_interval: 时间分割阈值，相邻两点超过特定值，此处为6h，以此分割轨迹段
    :param max_speed: 正常速度阈值，相邻两点平均速度超过设定值(此处为100Km/h)，删除异常速度的轨迹点
    :param gjd_list: 轨迹点列表
    :return:
    """
    if not gjd_list:
        return '传入列表为空'
    # 按照时间去重，存放所有时间戳索引的列表
    timestamp_list = []
    # 重复时间的索引列表，保留出现的第一个时间戳，把后续的删除(暂时删除，输出出来)
    repeat_timestamp = []
    for gjd_index, gjd in enumerate(gjd_list):
        # 拿到每个轨迹点信息和对应索引，进行分别处理
        gjd_timestamp = gjd[1]  # 对应轨迹点的时间戳
        if gjd_timestamp in timestamp_list:  # 如果当前时间戳已经在列表中存在，则该索引以及前一个索引对应的时间戳重复
            # print(gjd_index, '与上一个重复,将索引加入到列表中')
            repeat_timestamp.append(gjd_index)
            # 后续考虑是否删除
            # print('删除', gjd_index, '的轨迹点')
        timestamp_list.append(gjd_timestamp)  # 每个时间戳放到列表中，可以判断是否存在重复

    normal_time_list = [gjd_list[i] for i in range(len(gjd_list)) if (i not in repeat_timestamp)]  # 包含正常时间的轨迹点
    # 处理跳点(按照相邻两点的平均速度)
    current_index = 0
    running_periods = []
    current_running_period = []
    data_list = []
    while current_index < len(normal_time_list):
        current_point = normal_time_list[current_index]  # 当前索引对应的轨迹点
        lon = float(current_point[3])
        lat = float(current_point[2])
        timestamp = int(current_point[1])
        next_index = current_index + 1
        if next_index < len(normal_time_list):
            next_point = normal_time_list[next_index]  # 当前点的下一轨迹点
            next_lon = float(next_point[3])
            next_lat = float(next_point[2])
            next_timestamp = int(next_point[1])
            # 计算两点之间的距离
            distance = get_distance(lon, lat, next_lon, next_lat)  # 单位千米
            # 获取两点之间的时间差，将ms转换为h
            d_timestamp = (next_timestamp - timestamp) / 1000 / 3600
            # 如果d_timestamp!=0 d_timestamp=d_timestamp，如果=0则赋值为0.000001
            d_timestamp = d_timestamp if d_timestamp else 0.0000001
            # 如果时间差超过设定阈值，按照时间分割轨迹段
            if d_timestamp >= time_interval:
                current_running_period.append(normal_time_list[current_index])
                running_periods.append(current_running_period)
                current_running_period = []
                current_index += 1
                continue
            avg_speed = distance / d_timestamp  # 获取平均速度，单位 Km/h
            # 如果平均速度超过阈值，判断为跳点，跳过跳点，将当前轨迹点加入到航行段中
            if avg_speed >= max_speed:
                current_running_period.append(normal_time_list[current_index])
                current_index += 2
                continue
            else:
                current_running_period.append(normal_time_list[current_index])
                current_index += 1
        # 如果循环已经到达最后一个点，则直接将当前轨迹点添加
        elif next_index == len(normal_time_list):
            current_running_period.append(normal_time_list[current_index])
            running_periods.append(current_running_period)
            break
        #分割后若每条轨迹点大于10点，则当作有效轨迹保存
    for running_period in running_periods:
        if len(running_period) > 10:
            data_list.append(running_period)

    return data_list


def clear_jump_point_by_D_value(ship_points, D_value=10, default_speed=30):
    for ship, trajectory_list in tqdm(ship_points.items()):
        for points in trajectory_list:

            while len(points) > 0:
                if points[0][4] > default_speed:
                    del points[0]
                else:
                    break

            current_point_index = 0
            while current_point_index < len(points):
                next_point_index = current_point_index + 1
                if next_point_index <= len(points) - 1:
                    current_speed = points[current_point_index][4]
                    next_speed = points[next_point_index][4]
                    if next_speed - current_speed > D_value:
                        del points[next_point_index:len(points)]
                        continue
                    else:
                        current_point_index += 1
                else:
                    break
    return ship_points


def list_interpolation(list_datas=None, equal_time_interval=TIMESTAMP_interval,
                       last_time_interval=TIMESTAMP_difference):
    """
    对时间不均的轨迹点列表进行插值，按照设定的时间间隔进行插值
    插值函数选择三次样条函数
    :param list_datas: 轨迹点列表
    :param equal_time_interval:
    :param last_time_interval:
    :return: 返回插值好的新的轨迹点列表
    """
    #具有不均匀时间间隔 （running_period） 的轨迹点列表
    running_period = list_datas
    total_running_periods = []  # 对应船只航行段的总量列表
    new_running_period = []  # 新的航行段的信息
    timestamp_list = []  # 当作插值函数自变量

    new_timestamp_list = []  # 等时间间隔的时间戳列表
    lat_variable_list = []
    lon_variable_list = []
    # sog_variable_list = []
    # cog_variable_list = []
    variable_list = [lat_variable_list, lon_variable_list]  # 存放原始数据的因变量，暂定经纬度，需要通过插值获得
    # 将原始的字段对应的值加到原始因变量列表中
    for running_gjd in running_period:
        timestamp = int(running_gjd[1])
        timestamp_list.append(timestamp)  # 自变量列表
        lat = float(running_gjd[2])
        lat_variable_list.append(lat)
        lon = float(running_gjd[3])
        lon_variable_list.append(lon)
        # sog = float(running_gjd[4])
        # sog_variable_list.append(sog)
        # cog = float(running_gjd[5])
        # cog_variable_list.append(cog)
    first_timestamp = timestamp_list[0]  # 真实轨迹点第一个时间戳，也是插值后的第一个轨迹点的真实时间戳
    last_timestamp = timestamp_list[-1]  # 真实轨迹点最后一个时间戳
    range_number = int((last_timestamp - first_timestamp) / (60 * 1000)) + 3  # 获得大约循环次数数值
    # 按照第一个时间戳，进行后面等时间间隔的累加，加到与最后一个时间戳的差值在阈值范围
    for i in range(range_number):
        set_timestamp = first_timestamp + i * equal_time_interval
        new_timestamp_list.append(set_timestamp)
        if set_timestamp > last_timestamp and abs(set_timestamp - last_timestamp) < last_time_interval:
            # print(set_timestamp)
            break
    # print('----------------------------------')
    # 将时间戳列表减去第一个时间戳并除以1000 转换为函数可以接受的小范围(13位时间戳作为自变量组成的函数值超出浮点数范围)
    origin_independent_variable_list = []
    new_independent_variable_list = []
    for timestamps_0 in timestamp_list:
        origin_time = (timestamps_0 - first_timestamp) / 1000  # 转换成小数据格式
        origin_independent_variable_list.append(origin_time)
    for timestamps_1 in new_timestamp_list:
        new_time = (timestamps_1 - first_timestamp) / 1000  # 转换成小数据格式
        new_independent_variable_list.append(new_time)

    # 将时间自变量列表转换成数组，方便后续三次样条插值的输入
    origin_independent_variable_list = pd.Series(origin_independent_variable_list)
    new_independent_variable_list = pd.Series(new_independent_variable_list)
    # 插值法获得的新的因变量列表(lat,lon,sog,cog)
    new_lat_list = []
    new_lon_list = []
    # new_sog_list = []
    # new_cog_list = []
    # 获得对应索引的拉格朗日插值函数，按照列表variable_list所示0-lat,1-lon,2-sog,3-cog
    for index, dependent_variable in enumerate(variable_list):
        # fx = lagrange(origin_independent_variable_list, dependent_variable)  # 拉格朗日函数
        dependent_variable = pd.Series(dependent_variable)  # 样条函数
        fx = interpolate.splrep(origin_independent_variable_list, dependent_variable)  # 三次样条函数插值
        yy = interpolate.splev(new_independent_variable_list, fx, der=0)  # 获得对应新时间戳的自变量数组列表
        for i in yy:
            y_ = np.around(i, 5)  # 对numpy中的float64类型保留5位有效数字
            if index == 0:
                new_lat_list.append(y_)
            elif index == 1:
                new_lon_list.append(y_)
                # elif index == 2:
                #     new_sog_list.append(y_)
    new_times = []
    # origin_times = []
    # for origin_time in origin_independent_variable_list:
    #     origin_time = origin_time * 1000 + first_timestamp  # 重新变为13位时间戳的格式
    #     origin_times.append(origin_time)
    for new_time_ in new_independent_variable_list:
        new_time_ = new_time_ * 1000 + first_timestamp
        new_times.append(int(new_time_))
    # total_running_periods.append(new_running_period)
    for i in range(len(new_times)):
        new_list = [new_times[i], new_lat_list[i], new_lon_list[i]]
        total_running_periods.append(new_list)
    return total_running_periods


def list_segmentation(all_gjd_list=None, parking_dis=stop_DISTANCE, parking_length=parking_number):
    """
    对包含轨迹点的列表进行分割
    :param parking_length:
    :param parking_dis:
    :param all_gjd_list:
    :return:
    """
    parking_periods = []
    parking_index = []  # 停驻段索引列表

    for current_index, continuous_gjd in enumerate(all_gjd_list):
        current_lat = continuous_gjd[1]
        current_lon = continuous_gjd[2]
        next_index = current_index + 1
        if next_index < len(all_gjd_list):
            next_lat = all_gjd_list[next_index][1]
            next_lon = all_gjd_list[next_index][2]
            distance = get_distance(current_lon, current_lat, next_lon, next_lat)
            if distance <= parking_dis:  # 进入停驻段
                parking_index.append(current_index)
            else:  # 驶出停驻段，进入新的航行段
                if parking_index:
                    parking_periods.append(parking_index)
                    parking_index = []
                else:
                    continue
        elif next_index == len(all_gjd_list):
            if parking_index:
                parking_periods.append(parking_index)
            else:
                continue
    # print(1)
    running_periods = []  # 存放航行轨迹段
    running_index = []  # 存放每个航行轨迹段的始终索引
    # 通过停驻段判断航行段
    for index, parking_period in enumerate(parking_periods):
        # 遍历停驻段列表，如果当前停驻段列表长度大于设定阈值，判断为船舶停止状态，以此切断分割航行段
        if len(parking_period) > parking_length:
            # 单独判断第一个停驻段列表，观察全部轨迹点是以行驶状态开始还是停驻状态开始
            if index == 0:
                if parking_period[0] == 0:
                    # 如果从第一个轨迹点开始航速就为0，第一段航行段起始点为停驻段后一个点
                    running_start_index = parking_period[-1] + 1
                    running_index.append(running_start_index)  # 将航行段起始索引加入到航行轨迹段的索引列表
                else:
                    # 如果第一个轨迹点开始航速不为0，航行段索引值从0开始，以第一个停驻段前一个索引值为结束
                    running_start_index = 0
                    running_end_index = parking_period[0] - 1
                    running_index = [running_start_index, running_end_index]  # 得到一条航行轨迹段，根据索引值确定
                    running_periods.append(running_index)  # 将该航行段加入到航行段列表中
                    running_index = []  # 航行段索引列表清空，预计存放下一个航行段
            elif running_index:
                # 如果航行段索引列表存在，说明只有航行段起始索引，填入航行段结束索引
                running_end_index = parking_period[0] - 1
                running_index.append(running_end_index)
                running_periods.append(running_index)
                running_index = []
            else:
                # 航行段索引为空，判断当前停驻段后索引轨迹点为航行段起点，加到航向段索引列表中
                running_start_index = parking_period[-1] + 1
                running_index.append(running_start_index)
    # 如果循环停驻段结束后，航行段索引列表仍不为空，证明最后一段停驻段后仍在航行，直到整段轨迹段结束，将所有轨迹点最后一个轨迹点索引加入
    if len(running_index) == 1:
        running_index.append(len(all_gjd_list) - 1)
        running_periods.append(running_index)

    # 將航行段索引对应的轨迹点加入列表返回
    running_gjd_periods = []

    for running_index_period in running_periods:
        running_period = []
        running_gjd_period = all_gjd_list[running_index_period[0]:running_index_period[1] + 1]
        # 对按照停驻段分割的航行段按照时间间隔阈值进行分割
        for running_gjd_index, every_gjd in enumerate(running_gjd_period):
            running_period.append(every_gjd)
        running_gjd_periods.append(running_period)

    return running_gjd_periods


def get_distance(lon1, lat1, lon2, lat2):
    """
    计算经纬度之间的距离，单位千米
    :param lon1: A点的经度
    :param lat1: A点的纬度
    :param lon2: B点的经度
    :param lat2: B点的纬度
    :return:
    """
    lon1, lat1, lon2, lat2 = map(math.radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    d_lon = lon2 - lon1
    d_lat = lat2 - lat1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    distance = c * EARTH_RADIUS
    return distance


def get_course(lonA, latA, lonB, latB):
    """
    point p1(latA, lonA)
    point p2(latB, lonB)
    根据两点经纬度计算方向角，默认北半球
    :param latA:
    :param lonA:
    :param latB:
    :param lonB:
    :return:
    """
    radLonA = math.radians(lonA)
    radLatA = math.radians(latA)
    radLonB = math.radians(lonB)
    radLatB = math.radians(latB)
    dLon = radLonB - radLonA
    y = math.sin(dLon) * math.cos(radLatB)
    x = math.cos(radLatA) * math.sin(radLatB) - math.sin(radLatA) * math.cos(radLatB) * math.cos(dLon)
    brng = math.degrees(math.atan2(y, x))
    brng = (brng + 360) % 360
    return (round(brng) / 360) * math.pi


def get_speed(timeA, lonA, latA, timeB, lonB, latB):
    """
    传入两点的时间和经纬度信息
    时间为13位时间戳信息：ms→h
    :param timeA:
    :param lonA:
    :param latA:
    :param timeB:
    :param lonB:
    :param latB:
    :return:
    """
    a_time = int(timeA)
    b_time = int(timeB)
    a_lon = float(lonA)
    a_lat = float(latA)
    b_lon = float(lonB)
    b_lat = float(latB)
    d_timestamp = (b_time - a_time) / 1000 / 3600
    distance = get_distance(a_lon, a_lat, b_lon, b_lat)
    speed = distance / d_timestamp  # 获取平均速度，单位 Km/h
    # 千米每小时转化为米每秒
    return speed / 36


def normalization(data=None, min_=None, max_=None):
    """
    对传入参数列表进行归一化操作
    :param max_:
    :param min_:
    :param data:
    :return:
    """
    data = float(data)
    new_a = (data - min_) / (max_ - min_)
    return new_a


def get_derivative(f, delta=1e-10):
    """导函数生成器"""

    def derivative(x):
        """导函数"""
        return (f(x + delta) - f(x)) / delta

    return derivative


if __name__ == '__main__':
    # data_preprocessed = preprocessing(path=PATH_AIS_TEST_TXT)

    for root, dirs, files in os.walk(r"../data"):
        # root	表示正在遍历的文件夹的名字（根/子）
        # dirs	记录正在遍历的文件夹下的子文件夹集合
        # files	记录正在遍历的文件夹中的文件集合
        for file in tqdm(files):
            path = os.path.join(root, file)
            file_name = file.replace('.csv', '')
            ref_classify_dict = {}
            with open(path) as f:
                a = csv.DictReader(f)
                for row in a:
                    mmsi = row["MMSI"]
                    basetime = row["BaseDateTime"]
                    basetime = basetime.replace('T', ' ')
                    basetime = time.strptime(basetime, "%Y-%m-%d %H:%M:%S")
                    timestamp = int(time.mktime(basetime)) * 1000  # 转换为13位时间戳，单位为毫秒
                    list_row_data = [row["MMSI"], timestamp, row["LAT"], row["LON"], row["SOG"], row["COG"],
                                     row["Heading"], row["VesselType"], row["Length"], row["Width"], row["Draft"]]
                    if mmsi in ref_classify_dict.keys():
                        ref_classify_dict[mmsi].append(list_row_data)
                    else:
                        ref_classify_dict[mmsi] = [list_row_data]
                    # ques=ref_classify_dict.keys()
                #   preprocessing():
                #   每个mmsi，先排序，按时间间隔分割好，对每个分割好的轨迹进行插值和停驻段的判断和分割，拿到每个航迹段,
                #   在原有[时间戳，lat，lon]的基础上添加数据进行完善
                #   [时间戳，lat，lon, 角度，速度， 航行距离，船类型， 月，日，小时，开始轨迹点的lat，开始轨迹点的lon，mmsi]
                data_preprocessed = preprocessing(ref_classify_dict)
                with open('../dateset/' + file_name + '.json', 'w', encoding='utf-8') as f:
                    json.dump(data_preprocessed, f)
    gen_dataset()
