import torch
from torch import nn

from preprocess.conf import len_label

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        pred = self.linear(output)
        # 取最后一个时间步的输出作为LSTM网络的输出
        pred = pred[:, -1, :]
        return pred


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_seq):
        h_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, self.batch_size, self.hidden_size).to(device)
        # print(input_seq.size())
        seq_len = input_seq.shape[1]
        # input(batch_size, seq_len, input_size)
        input_seq = input_seq.view(self.batch_size, seq_len, self.input_size)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
       #  output(, , 0,)表示前隐藏状态  output(, , 1,)表示后隐藏状态
        output = output.contiguous().view(self.batch_size, seq_len, self.num_directions, self.hidden_size)
        # 将前后隐藏状态用均值方式合并，这时output维度变为(batch_size, seq_len,  hidden_size)
        output = torch.mean(output, dim=2)
        pred = self.linear(output)
        # print('pred=', pred.shape)
        pred = pred[:, -1, :]

        return pred


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        # output, (h, c) = self.lstm(input_seq, (h_0, c_0))
        # output, h, c = self.lstm(input_seq, h_0, c_0)

        return output, h, c
        # return output, h


class Decoder(nn.Module):
    def __init__(self,  hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.input_size = output_size
        self.hidden_size = hidden_size
        #   文章中指定的lstm为五层
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 2
        self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers * self.num_directions, batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 8代表舰船类型特征维度是8   8+6为ort
        self._linear = nn.Linear(8 + 6 + self.input_size, self.input_size)


    def forward(self, input_seq, h, c, type_emb):
        # input_seq(batch_size, input_size)
        batch_size = input_seq.shape[0]
        input_seq = input_seq.view(batch_size, 1, self.input_size)
        type_emb = type_emb.view(batch_size, 1, -1)
        # input_seq(128,1,16)
        input_seq = torch.cat((input_seq, type_emb), dim=2)
        # input_seq(128,1,2)
        input_seq = torch.tanh(self._linear(input_seq))
        output, (h, c) = self.lstm(input_seq, (h, c))
        # output, h = self.lstm(input_seq, h)
        # output(batch_size, seq_len, num * hidden_size)
        pred = self.linear(output)  # pred(batch_size, 1, output_size)
        pred = pred[:, -1, :]

        return pred, h, c
        # return pred, h
        # return output, h, c


# 原有的Seq2Seq结构
class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super().__init__()
        self.output_size = output_size
        self.batch_size = batch_size
        self.Encoder = Encoder(input_size, hidden_size, num_layers, batch_size)
        self.Decoder = Decoder(hidden_size, num_layers, output_size, batch_size)

    def forward(self, input_seq, label, ship_type_emb, Train=True):
        ship_type_emb = ship_type_emb.detach()
        batch_size, seq_len, input_size = input_seq.shape[0], input_seq.shape[1], input_seq.shape[2]
        output, h, c = self.Encoder(input_seq)
        # output, h = self.Encoder(input_seq)

        pred_num = len_label
        outputs = torch.zeros(batch_size, pred_num, self.output_size).to(device)

        output = torch.zeros(batch_size, self.output_size).to(device)
        # for t in self.output_size:
        # output[:,0]=input_seq[:,-1,1]
        # output[:,1]=input_seq[:,-1,5]
        # output[:,2]=input_seq[:,-1,6]
        if Train:
            for t in range(pred_num):
                input = input_seq[:, -1, :2] if t == 0 else label[:, t - 1, :]
                # _input = input_seq[:, -1, :]
                # output, h, c = self.Decoder(input, h, c)
                # TODO: ship_type_emb[:, t, :]代表的就是训练10个点的前5个点的ship_type_emb ？？
                ship_type_emb_step = ship_type_emb[:, t, :]
                output, h, c = self.Decoder(input, h, c, ship_type_emb_step)
                # output.view(batch_size, input_size)
                outputs[:, t, :] = output
                # output.view(batch_size, 1, self.input_size)
        else:
            # 第一个值用训练的最后一个值，后面每次都用预测产生的值
            output = input_seq[:, -1, :2]
            for t in range(pred_num):
                ship_type_emb_step = ship_type_emb[:, t, :]
                output, h, c = self.Decoder(output, h, c, ship_type_emb_step)
                # output.view(batch_size, input_size)
                outputs[:, t, :] = output
                # output.view(batch_size, 1, self.input_size)
        # return outputs[:, -1, :]
        return outputs


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class MAE(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        criterion = nn.L1Loss()
        loss = criterion(x, y)
        return loss
