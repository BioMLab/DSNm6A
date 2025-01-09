import torch.nn as nn
from function import ReverseLayerF
import torch


class model_1(nn.Module):

    def __init__(self):
        super(model_1, self).__init__()
        self.encoder_conv = nn.Sequential()

        self.encoder_conv.add_module('pse_flatten2', nn.Flatten(start_dim=2))
        self.encoder_conv.add_module('pse_conv1', nn.Conv1d(1, 32, 28, padding=14))
        self.encoder_conv.add_module('pse_pool1', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pse_RelU1', nn.ReLU())
        self.encoder_conv.add_module('pse_BN1', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout1', nn.Dropout(0.2))

        self.encoder_conv.add_module('pse_conv2', nn.Conv1d(32, 32, 28, padding=14))
        self.encoder_conv.add_module('pse_pool2', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pse_RelU2', nn.ReLU())
        self.encoder_conv.add_module('pse_BN2', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout2', nn.Dropout(0.2))

        self.encoder_conv.add_module('pse_conv3', nn.Conv1d(32, 32, 28, padding=14))
        self.encoder_conv.add_module('pse_pool3', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pse_RelU3', nn.ReLU())
        self.encoder_conv.add_module('pse_BN3', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout3', nn.Dropout(0.2))

        self.encoder_conv.add_module('pse_conv4', nn.Conv1d(32, 32, 28, padding=14))
        self.encoder_conv.add_module('pse_pool4', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pse_RelU4', nn.ReLU())
        self.encoder_conv.add_module('pse_BN4', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout4', nn.Dropout(0.2))


        self.encoder_gru = nn.Sequential()
        self.encoder_gru.add_module('pse_gru1', nn.LSTM(70, 32*2, batch_first=True, bidirectional=True))
        self.encoder_bn = nn.Sequential()
        self.encoder_bn.add_module('pse_BN', nn.BatchNorm1d(32))
        self.encoder_bn.add_module('pse_flatten2', nn.Flatten(start_dim=1))

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.add_module('pse_Linear1', nn.Linear(4096, 1024))
        self.encoder_fc.add_module('pse_relu1', nn.ReLU(True))
        self.encoder_fc.add_module('pse_dropout1', nn.Dropout(p=0.2))
        self.encoder_fc.add_module('pse_Linear2', nn.Linear(1024, 256))


    def forward(self, input_data):
        input_data = input_data.reshape(len(input_data), 1, -1)
        feat = self.encoder_conv(input_data)
        # print(feat.shape)
        feat, _ = self.encoder_gru(feat)
        feat = self.encoder_bn(feat)
        code = self.encoder_fc(feat)

        return code


class CG(nn.Module):

    def __init__(self):
        super(CG, self).__init__()
        self.encoder_conv = nn.Sequential()
        self.encoder_conv.add_module('pse_conv1', nn.Conv1d(32, 32, 1))
        self.encoder_conv.add_module('pse_RelU1', nn.ReLU())
        self.encoder_conv.add_module('pse_BN1', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout1', nn.Dropout(0.2))

        self.encoder_conv.add_module('pse_conv2', nn.Conv1d(32, 32, 1))
        self.encoder_conv.add_module('pse_RelU2', nn.ReLU())
        self.encoder_conv.add_module('pse_BN2', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout2', nn.Dropout(0.2))

        self.encoder_conv.add_module('pse_flatt', nn.Flatten(start_dim=1))
        self.encoder_conv.add_module('Linear', nn.Linear(32, 32))
        self.encoder_conv.add_module('pse_RelU3', nn.ReLU(True))

        self.encoder_lstm = nn.Sequential()
        self.encoder_lstm.add_module('lstm_flatt', nn.Flatten(start_dim=1))
        # self.encoder_lstm.add_module('lstm', nn.LSTM(8, 8))
        self.encoder_lstm.add_module('lstm', nn.LSTM(32, 16, batch_first=True, bidirectional=True))

        self.encoder_relu = nn.Sequential()
        self.encoder_relu.add_module('pse_relu4', nn.ReLU(True))

    def forward(self, input_data):
        # print(type(input_data))
        code1 = self.encoder_conv(input_data)
        code2, _ = self.encoder_lstm(input_data)
        code = code1 + code2
        code = self.encoder_relu(code).reshape(-1, 32, 1)
        return code


class model_2(nn.Module):

    def __init__(self):
        super(model_2, self).__init__()
        self.encoder_conv = nn.Sequential()
        self.encoder_conv.add_module('pse_conv1', nn.Conv1d(328, 32, 1, padding=0))
        self.encoder_conv.add_module('pse_RelU1', nn.ReLU())
        self.encoder_conv.add_module('pse_BN1', nn.BatchNorm1d(32))
        self.encoder_conv.add_module('pse_dropout1', nn.Dropout(0.2))

        # self.encoder_conv.add_module('pse_conv2', nn.Conv1d(64, 32, 1, padding=0))
        # self.encoder_conv.add_module('pse_RelU2', nn.ReLU())
        # self.encoder_conv.add_module('pse_BN2', nn.BatchNorm1d(32))
        # self.encoder_conv.add_module('pse_dropout2', nn.Dropout(0.2))

        self.encoder_CG1 = nn.Sequential()
        self.encoder_CG1.add_module('pse_CG1', CG())

        self.encoder_CG2 = nn.Sequential()
        self.encoder_CG2.add_module('pse_CG2', CG())

        self.encoder_CG3 = nn.Sequential()
        self.encoder_CG3.add_module('pse_CG3', CG())

        self.encoder_CG4 = nn.Sequential()
        self.encoder_CG4.add_module('pse_CG4', CG())

        self.encoder_CG5 = nn.Sequential()
        self.encoder_CG5.add_module('pse_CG5', CG())

        self.encoder_conv1 = nn.Sequential()
        self.encoder_conv1.add_module('pse_conv3', nn.Conv1d(32, 32, 1, padding=0))
        self.encoder_conv1.add_module('pse_RelU3', nn.ReLU())
        self.encoder_conv1.add_module('pse_BN3', nn.BatchNorm1d(32))
        self.encoder_conv1.add_module('pse_dropout3', nn.Dropout(0.2))
        self.encoder_conv1.add_module('pse_flatt', nn.Flatten(start_dim=1))
        self.encoder_conv1.add_module('pse_linear', nn.Linear(32, 50))

    def forward(self, input_data):
        input_data = input_data.reshape(len(input_data), -1, 1)
        code = self.encoder_conv(input_data)
        code = self.encoder_CG1(code)
        # code = self.encoder_CG2(code)
        # code = self.encoder_CG3(code)
        # code = self.encoder_CG4(code)
        # code = self.encoder_CG5(code)
        code = self.encoder_conv1(code)
        return code


class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        self.encoder_conv = nn.Sequential()
        self.encoder_conv.add_module('pte_PCP_pskp_conv1',
                                     nn.Conv1d(in_channels=1, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool1', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN1', nn.BatchNorm1d(6))

        self.encoder_conv.add_module('pte_PCP_pskp_conv2',
                                     nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool2', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN2', nn.BatchNorm1d(6))

        self.encoder_conv.add_module('pte_PCP_pskp_conv3',
                                     nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool3', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN3', nn.BatchNorm1d(6))

        self.encoder_conv.add_module('pte_PCP_pskp_conv4',
                                     nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool4', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN4', nn.BatchNorm1d(6))

        self.encoder_conv.add_module('pte_PCP_pskp_conv5',
                                     nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool5', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN5', nn.BatchNorm1d(6))

        self.encoder_conv.add_module('pte_PCP_pskp_conv6',
                                     nn.Conv1d(in_channels=6, out_channels=6, kernel_size=10, padding=5))
        self.encoder_conv.add_module('pte_PCP_pskp_pool6', nn.MaxPool1d(2))
        self.encoder_conv.add_module('pte_PCP_pskp_BN6', nn.BatchNorm1d(6))

        self.encoder_gru = nn.Sequential()
        self.encoder_gru.add_module('pte_PCP_pskp_gru1', nn.LSTM(9, 16, batch_first=True, bidirectional=True))

        self.encoder_bn = nn.Sequential()
        self.encoder_bn.add_module('pte_PCP_pskp_BN', nn.BatchNorm1d(6))
        self.encoder_bn.add_module('pte_PCP_pskp_flatten2', nn.Flatten(start_dim=1))

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.add_module('pte_PCP_pskp_relu1', nn.ReLU())
        self.encoder_fc.add_module('pte_PCP_pskp_Linear1', nn.Linear(192, 1024))
        self.encoder_fc.add_module('pte_PCP_pskp_relu2', nn.ReLU())
        self.encoder_fc.add_module('pse_dropout1', nn.Dropout(p=0.2))
        self.encoder_fc.add_module('pte_PCP_pskp_Linear2', nn.Linear(1024, 256))
        self.encoder_fc.add_module('pte_PCP_pskp_relu3', nn.ReLU())
        self.encoder_fc.add_module('pse_dropout2', nn.Dropout(p=0.2))
        self.encoder_fc.add_module('pte_PCP_pskp_Linear3', nn.Linear(256, 50))

    def forward(self, input_data):
        input_data = input_data.reshape(len(input_data), 1, -1)
        code = self.encoder_conv(input_data)
        code, _ = self.encoder_gru(code)
        code = self.encoder_bn(code)
        code = self.encoder_fc(code)
        return code
















