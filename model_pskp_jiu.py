import torch.nn as nn
from function import ReverseLayerF
import torch
from model_full import model_1, model_3
from BERT import BERT
import random
import numpy as np

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class DSN(nn.Module):
    def __init__(self, code_size=50, n_class=1):
        super(DSN, self).__init__()
        # [[1, 520.0], [2, 260.0], [4, 130.0], [5, 104.0], [8, 65.0], [10, 52.0], [13, 40.0], [20, 26.0],
        # [26, 20.0], [40, 13.0], [52, 10.0], [65, 8.0], [104, 5.0], [130, 4.0], [260, 2.0], [520, 1.0]]

        # 256, 512, 1024, 2048,
        self.code_size = code_size

        self.source_encoder_bert_pskp_PCP = nn.Sequential()
        self.source_encoder_bert_pskp_PCP.add_module('source_cnn', model_1())
        self.source_encoder_bert_pskp_PCP.add_module('source_bert', BERT())
        self.target_encoder_bert_pskp_PCP = nn.Sequential()
        self.target_encoder_bert_pskp_PCP.add_module('target_cnn', model_1())
        self.target_encoder_bert_pskp_PCP.add_module('target_bert', BERT())
        self.shared_encoder_bert_pskp_PCP = nn.Sequential()
        self.shared_encoder_bert_pskp_PCP.add_module('shared_cnn', model_1())
        self.shared_encoder_bert_pskp_PCP.add_module('shared_bert', BERT())


        self.shared_encoder_pred_domain = nn.Sequential()
        self.shared_encoder_pred_domain.add_module('fc_se1', nn.Linear(in_features=50, out_features=16))
        self.shared_encoder_pred_domain.add_module('relu_se1', nn.ReLU(True))
        self.shared_encoder_pred_domain.add_module('fc_se12', nn.Linear(in_features=16, out_features=1))


        self.shared_encoder_pred_class = nn.Sequential()
        self.shared_encoder_pred_class.add_module('fc_se21', nn.Linear(in_features=code_size, out_features=16))
        self.shared_encoder_pred_class.add_module('relu_se2', nn.ReLU(True))
        self.shared_encoder_pred_class.add_module('fc_se22', nn.Linear(in_features=16, out_features=n_class))


        self.shared_decoder_fc_pskp_PCP = nn.Sequential()
        self.shared_decoder_fc_pskp_PCP.add_module('fc_sd3', nn.Linear(in_features=code_size, out_features=1108))
        self.shared_decoder_fc_pskp_PCP.add_module('relu_sd3', nn.ReLU(True))
        self.shared_decoder_fc_pskp_PCP.add_module('p_conv1',
                                                   nn.Conv1d(in_channels=1, out_channels=6, kernel_size=11, padding=5))
        self.shared_decoder_fc_pskp_PCP.add_module('p_batch1', nn.BatchNorm1d(6))
        self.shared_decoder_fc_pskp_PCP.add_module('p_conv5',
                                                   nn.Conv1d(in_channels=6, out_channels=1, kernel_size=11, padding=5))
        self.shared_decoder_fc_pskp_PCP.add_module('p_true5', nn.ReLU(True))



        # self.alpha_weight = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.beta_weight = nn.Parameter(torch.rand(1)*0.5, requires_grad=True)
        # self.gamma_weight = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.class_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # self.alpha_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
        # self.beta_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
        # self.gamma_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
        # self.class_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)


        self.softmax_weight = nn.Softmax(dim=0)






    def forward(self, input_data_pskp_PCP, mode, rec_scheme, p=0.0):

        result = []
        if mode == 'source':
            private_code_bert_pskp_PCP = self.source_encoder_bert_pskp_PCP(input_data_pskp_PCP)
        elif mode == 'target':
            private_code_bert_pskp_PCP = self.target_encoder_bert_pskp_PCP(input_data_pskp_PCP)
        result.append(private_code_bert_pskp_PCP)
        shared_code_bert_pskp_PCP = self.shared_encoder_bert_pskp_PCP(input_data_pskp_PCP)
        result.append(shared_code_bert_pskp_PCP)
        shared_code = shared_code_bert_pskp_PCP
        reversed_shared_code = ReverseLayerF.apply(shared_code, p)

        # print("reversed_shared_code", reversed_shared_code)
        # print("reversed_shared_code", reversed_shared_code.shape)
        domain_label = self.shared_encoder_pred_domain(reversed_shared_code)
        result.append(domain_label)
        class_shared_label = torch.sigmoid(self.shared_encoder_pred_class(shared_code))
        class_label = 1 * class_shared_label
        result.append(class_label)

        if rec_scheme == 'share':
            union_pskp_PCP_code = shared_code_bert_pskp_PCP
        elif rec_scheme == 'all':
            union_pskp_PCP_code = private_code_bert_pskp_PCP + shared_code_bert_pskp_PCP
        elif rec_scheme == 'private':
            union_pskp_PCP_code = private_code_bert_pskp_PCP

        union_pskp_PCP_code = union_pskp_PCP_code.view(-1, 1, 50)
        rec_pskp_PCP_vec = self.shared_decoder_fc_pskp_PCP(union_pskp_PCP_code)
        rec_pskp_PCP_vec = rec_pskp_PCP_vec.view(-1, 1, 1108)

        # self.softmax_weight(self.alpha_weight, self.beta_weight, self.gamma_weight, self.class_weight)

        result.append(rec_pskp_PCP_vec)
        return result

class Params(nn.Module):
    def __init__(self):
        super(Params, self).__init__()
        # self.alpha_weight = nn.Parameter(torch.randn(1), requires_grad=True)
        # self.beta_weight = nn.Parameter(torch.randn(1)*0.2, requires_grad=True)
        # self.gamma_weight = nn.Parameter(torch.randn(1), requires_grad=True)
        # # self.alpha_weight_linear = nn.Linear(1, 1)
        # # self.beta_weight_linear = nn.Linear(1, 1)
        # # self.gamma_weight_linear = nn.Linear(1, 1)
        # # self.alpha_weight = torch.tensor([0.02])
        # # self.beta_weight = torch.tensor([0.05])
        # # self.gamma_weight = torch.tensor([0.25])
        # self.fc1 = nn.Linear(3, 3)  # 全连接层
        # self.softmax = nn.Softmax(dim=1)  # softmax层
        # self.fc2 = nn.Linear(3, 3)  # 第二个全连接层



        self.fc1 = nn.Linear(3, 1)
        self.fc2 = nn.Linear(1, 1)
        self.softmax = nn.Softmax(dim=0)
        self.alpha_weight = nn.Parameter(torch.randn(3))
        self.beta_weight = nn.Parameter(torch.randn(3) * 0.5)
        self.gamma_weight = nn.Parameter(torch.randn(3))

    def forward(self):
        # # self.alpha_weight = self.alpha_weight_linear(self.alpha_weight)
        # # self.beta_weight = self.beta_weight_linear(self.beta_weight)
        # # self.gamma_weight = self.gamma_weight_linear(self.gamma_weight)
        # # 将三个权重参数组合成一个张量
        # weights = torch.cat([self.alpha_weight, self.beta_weight, self.gamma_weight], dim=0).unsqueeze(0)
        # weights = self.fc1(weights)
        # weights = self.softmax(weights)*0.5
        # weights = self.fc2(weights)
        # weights = self.softmax(weights) * 0.35
        # return weights.squeeze(0)

        weights = torch.stack([self.alpha_weight, self.beta_weight, self.gamma_weight], dim=1)
        weights = self.softmax(self.fc2(self.fc1(weights))) * 0.5
        # print("================" + str(weights))
        return weights



# class Params(nn.Module):
#     def __init__(self):
#         super(Params, self).__init__()
#         # self.alpha_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
#         # self.beta_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
#         # self.gamma_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
#         # self.class_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
#
#         # self.linear1 = nn.Linear(3, 3)
#         # self.soft = nn.Softmax()
#         # self.linear2 = nn.Linear(3, 3)
#
#
#         self.param = nn.Sequential()
#         self.param.add_module('fc_se1', nn.Linear(in_features=3, out_features=3))
#         self.param.add_module('fc_se2', nn.Softmax())
#         # self.param.add_module('fc_se3', nn.Linear(in_features=3, out_features=3))
#
#     def forward(self, alpha_weight, beta_weight, gamma_weight):
#         # print(alpha_weight)
#         # print(beta_weight)
#         # print(gamma_weight)
#         alpha_weight = torch.unsqueeze(alpha_weight, dim=0)
#         beta_weight = torch.unsqueeze(beta_weight, dim=0)
#         gamma_weight = torch.unsqueeze(gamma_weight, dim=0)
#         param = torch.cat((alpha_weight, beta_weight, gamma_weight),dim=1)
#         # print(param)
#         # param = torch.unsqueeze(param, dim=0)
#         # print("param", param)
#         # print(param)
#         # param = self.linear1(param)
#         # print(param)
#         # param = self.soft(param) * 0.5
#         # print(param)
#         # param = self.linear2(param)
#
#
#         param = self.param(param) * 0.5
#         # print(param)
#
#
#         return param



