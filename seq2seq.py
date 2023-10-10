# -*- coding:utf-8 -*-
"""
@Time：2022/05/25 23:20
@Author：KI
@File：seq2seq.py
@Motto：Hungry And Humble
"""
import os
import sys
from argparse import Namespace
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

path = os.path.abspath(os.path.dirname(os.getcwd()))
LSTM_PATH = path + '/models/seq2seq.pkl'

from preprocess.conf import len_train, len_label, BATCH_SIZE
from preprocess.feature_engineering import gen_dataloader
from util import train, test



if __name__ == '__main__':
    # args = seq2seq_args_parser()
    flag = 'seq2seq'
    # flag = 'ms'
    args = Namespace(epochs=3, batch_size=BATCH_SIZE, bidirectional=True, file_name=os.path.join(path, 'data', 'ship_data.json'),
                     hidden_size=64, optimizer='adam', lr=0.001, weight_decay=0.0,
                     input_size=5, flag=flag, num_layers=5, step_size=100, gamma=0.1,
                     output_size=2, predict_num=len_label,
                     seq_len=len_train)
    # Dtr, Val, Dte, m, n = nn_seq(args)
    Dtr, Val, Dte, m, n = gen_dataloader(args)   #Dtr:train_data
    train(args, Dtr, Val, LSTM_PATH)
    test(args, Dte, LSTM_PATH, m, n)
