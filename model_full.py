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


