from torch.autograd import Function
import torch.nn as nn
import torch
import torch.nn.functional as F


# 反转层
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

# 差值的平方 / n   ,,,,,,,,,,,,均方差,,,,最后是一个数字的张量
class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred) # 张量相加
        n = torch.numel(diffs.data) # 可以得知tensor中一共包含多少个元素
        mse = torch.sum(diffs.pow(2)) / n # 对输入的tensor数据的某一维度求和
        return mse

# 差值的平方 / n的平方
class SIMSE(nn.Module):

    def __init__(self):
        super(SIMSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, - pred)
        n = torch.numel(diffs.data) # 可以得知tensor中一共包含多少个元素
        simse = torch.sum(diffs).pow(2) / (n ** 2) # 表示n的平方
        return simse

# 差异损失，，，，求预测的范数，预测 / 范数；；；；标签同样处理；；；；；二者矩阵相乘的平方是差异损失
class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6) # .expand_as(input2)将张量扩展为和参数input2一样的大小。

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))


        return diff_loss

#
# #计算Cell特征向量相似性(余选相似度)
# x_cell_sim = x_cell / torch.norm(x_cell, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
# cell_similarity = torch.mm(x_cell_sim, x_cell_sim.T)  # 矩阵乘法
# cell_similarity_np = cell_similarity.detach().cpu().numpy()

