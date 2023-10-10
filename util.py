import copy
from itertools import chain
import pandas as pd
import numpy as np

import torch
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

from models import LSTM, BiLSTM, Seq2Seq, RMSELoss, MAE
from preprocess.feature_engineering import Input_Module

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def get_val_loss(args, model, Val):
    label = []
    val_loss = []
    #   将模型设置为评估模式
    model.eval()
    loss_function = RMSELoss().to(device)
    print('validating...')
    val_total_time = 0
    input_module = Input_Module()
    for (seq, target) in tqdm(Val):
        currt_time = time.time()
        seq, target, type_data = input_module(seq, target)
        seq = seq.to(device)
        target = target.to(device)
        type_data = type_data.to(device)
        with torch.no_grad():
            y_pred = model(seq, target, type_data, False)
            loss = loss_function(y_pred, target)
            val_loss.append(loss.item())
        next_time = time.time()
        use_time = next_time - currt_time
        val_total_time += use_time
    return np.mean(val_loss), val_total_time


# 计算MAPE指标的函数
def get_mape(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def load_model(args):
    input_size, hidden_size, num_layers = args.input_size, args.hidden_size, args.num_layers
    output_size = args.output_size
    if args.flag in ['us', 'ms', 'mm']:
        if args.bidirectional:
            model = BiLSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
        else:
            model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    elif args.flag in ['seq2seq']:
        model = Seq2Seq(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    else:
        model = LSTM(input_size, hidden_size, num_layers, output_size, batch_size=args.batch_size).to(device)
    return model


def train(args, Dtr, Val, path):
    model = load_model(args)
    loss_function = RMSELoss().to(device)
    #   optimizer是优化器，用来更新模型参数，作用是最小化损失函数
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    # scheduler为学习率调度器 step_size是学习率衰减的步数 gamma是学习率衰减的比例
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # training
    min_epochs = -1
    best_model = None
    min_val_loss = 5
    best_loss = 5
    best_epoch = 0
    train_total_time = 0
    val_total_time = 0
    for epoch in tqdm(range(args.epochs)):
        train_loss = []
        input_module = Input_Module()
        #   seq:(128,10,13) label:(128,5,13)
        for step, (seq, label) in enumerate(Dtr):
            #   返回当前时间的时间戳
            current_time = time.time()
            # print('seq:', seq.size())
            # print('label:', label.size())
            # print()
            #   输出后seq:(128,10,5) label:(128,5,2) type_data:(128,10,14)
            seq, label, type_data = input_module(seq, label)    #TODO:   input_module部分的三个输出有疑问
            # print('---------------------------------------')
            # print()
            # print('seq:', seq.size())
            # print('label:', label.size())
            # print('type_data:', type_data.size())
            seq = seq.to(device)
            label = label.to(device)
            type_data = type_data.to(device)
            y_pred = model(seq, label, type_data)
            # 求loss
            loss = loss_function(y_pred, label)
            # loss.item()用来取loss的值，精度比用索引来的高，在求损失函数等时我们一般用.item()
            train_loss.append(loss.item())
            # 梯度初始化为零
            optimizer.zero_grad()
            # 反向传播求梯度
            loss.backward()
            # 更新所有参数
            optimizer.step()
            next_time = time.time()
            ues_time = next_time - current_time
            train_total_time += ues_time
            step += 1
            if step % 100 == 0:
                print("epoch :{},step :{} ,Train loss: {}".format(epoch, step, np.sum(train_loss) / step))
        #   更新学习率
        scheduler.step()
        # validation
        val_loss, val_time = get_val_loss(args, model, Val)
        if epoch > min_epochs and val_loss < min_val_loss:
            min_val_loss = val_loss
            best_model = copy.deepcopy(model)
        val_total_time += val_time
        print('epoch {:03d} train_loss {:.8f} val_loss {:.8f}'.format(epoch, np.mean(train_loss), val_loss))
        model.train()

    state = {'models': best_model.state_dict()}
    print('best_model:{}, min_val_loss:{}, train time:{}, val time:{}'.format(best_model, min_val_loss,
                                                                              train_total_time / args.epochs,
                                                                             val_total_time / args.epochs))
    # 保存学习到的参数
    torch.save(state, path)


def test(args, Dte, path, m, n):
    print('输入任意字符开始测试：')
    a = input()
    # y = []
    test_loss = []
    mae_loss = []
    loss_t_dict = {}
    loss_dict = {}
    print('loading models...')
    loss_function = RMSELoss().to(device)
    mae_loss_function = MAE().to(device)
    model = load_model(args)
    model.load_state_dict(torch.load(path, map_location='cpu')['models'])
    model.eval()
    test_total_time = 0
    print('predicting...')
    test_total_time = 0
    input_module = Input_Module()
    for (seq, target) in tqdm(Dte):
        current_time = time.time()
        seq, target, type_data = input_module(seq, target)
        # target = list(chain.from_iterable(target.data.tolist()))
        type_data = type_data.to(device)
        # y.extend(target)
        seq = seq.to(device)
        target = target.to(device)
        seq_len = target.shape[1]
        with torch.no_grad():
            # print(seq.shape)
            # print(target.shape)
            # print(type_data.shape)
            y_pred = model(seq, target, type_data, False)
            for t in range(seq_len):
                if t not in loss_dict.keys():
                    loss_dict[t] = []
                loss_dict[t].append(loss_function(y_pred[:, t, :], target[:, t, :]).item())
                if t not in loss_t_dict.keys():
                    loss_t_dict[t] = []
                loss_t_dict[t].append(loss_function(y_pred[:, :t + 1, :], target[:, :t + 1, :]).item())
            loss = loss_function(y_pred, target)
            test_loss.append(loss.item())
            mae = mae_loss_function(y_pred, target)
            mae_loss.append(mae.item())
        next_time = time.time()
        ues_time = next_time - current_time
        test_total_time += ues_time
    for i, loss_ in loss_dict.items():
        print('predict the {} point mean loss is {}'.format(i + 1, np.mean(loss_)))
    for i, _loss in loss_t_dict.items():
        print('test predict the first {} points` mean loss is {}'.format(i + 1, np.mean(_loss)))
    print('test mean rmse is {}, mae is {}, test time is {}'.format(np.mean(test_loss), np.mean(mae_loss),
                                                                    test_total_time))
