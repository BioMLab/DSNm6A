# [n,41,4],[n,40,4], [n,41,4], [[[[[[[[n,120],[n,400]]]]]]]]
from bianma import onehot, twohot, ncpf, pskp, PCP, PseDNC
# from bianma import onehot
import torch.optim as optim
import pandas as pd
import numpy as np
import torch
import os
import random
import torch.utils.data
from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
batch_size = 64
n_epoch = 200
lr=0.001
momentum=0.9
weight_decay=1e-5

seed = 2
# 设置随机数种子
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model_root = 'model'
cuda = 1
cudnn.benchmark = True
lr = 0.005  # 学习率

# # kfold
# kf = KFold(n_splits=5, shuffle=True, random_state=0)
# m6A_atlas_exonic = open('hg38/lncRNA/zhong/lncRNA_pos_41.txt', 'r').readlines()
# m6A_atlas_intronic = open('hg38/lncRNA/zhong/lncRNA_neg_41.txt', 'r').readlines()
# z = 1
# for train_index, test_index in kf.split(m6A_atlas_exonic):
#     f1 = open(str('hg38/data/train/pos_train_41') + str('.txt'), 'w')
#     for i in train_index:
#         f1.write(m6A_atlas_exonic[i])
#     f2 = open(str('hg38/data/indepent/pos_test_41') + str('.txt'), 'w')
#     for i in test_index:
#         f2.write(m6A_atlas_exonic[i])
#     f1.close()
#     f2.close()
#     z += 1
# z = 1
# for train_index, test_index in kf.split(m6A_atlas_intronic):
#     f1 = open(str('hg38/data/train/neg_train_41') + str('.txt'), 'w')
#     for i in train_index:
#         f1.write(m6A_atlas_intronic[i])
#     f2 = open(str('hg38/data/indepent/neg_test_41') + str('.txt'), 'w')
#     for i in test_index:
#         f2.write(m6A_atlas_intronic[i])
#     f1.close()
#     f2.close()
#     z += 1


# kf = KFold(n_splits=5, shuffle=True, random_state=0)
# m6A_atlas_exonic = open('hg38/data/train/pos_train_41.txt', 'r').readlines()
# m6A_atlas_intronic = open('hg38/data/train/neg_train_41.txt', 'r').readlines()
# z = 1
# for train_index, test_index in kf.split(m6A_atlas_exonic):
#     f1 = open(str('hg38/kfold/train/fen/pos_train_') + str(z) + str('.txt'), 'w')
#     for i in train_index:
#         f1.write(m6A_atlas_exonic[i])
#     f2 = open(str('hg38/kfold/train/fen/pos_test_') + str(z) + str('.txt'), 'w')
#     for i in test_index:
#         f2.write(m6A_atlas_exonic[i])
#     f1.close()
#     f2.close()
#     z += 1
# z = 1
# for train_index, test_index in kf.split(m6A_atlas_intronic):
#     f1 = open(str('hg38/kfold/train/fen/neg_train_') + str(z) + str('.txt'), 'w')
#     for i in train_index:
#         f1.write(m6A_atlas_intronic[i])
#     f2 = open(str('hg38/kfold/train/fen/neg_test_') + str(z) + str('.txt'), 'w')
#     for i in test_index:
#         f2.write(m6A_atlas_intronic[i])
#     f1.close()
#     f2.close()
#     z += 1







# # onehot编码
# filePath = 'hg38/kfold/train/fen/'
# files = os.listdir(filePath)
# for i in files:
#     encode = np.array(onehot(filePath + i))
#     np.save('hg38/kfold/train/onehot/' + i[:-4], encode)
# 
# filePath = 'hg38/data/indepent/neg_test_41.txt'
# encode = np.array(onehot(filePath))
# np.save('hg38/kfold/indepent/onehot/neg_test_41', encode)
# filePath = 'hg38/data/indepent/pos_test_41.txt'
# encode = np.array(onehot(filePath))
# np.save('hg38/kfold/indepent/onehot/pos_test_41', encode)
# 
# filePath = 'hg38/mRNA/zhong/mRNA_pos_41.txt'
# encode = np.array(onehot(filePath))
# np.save('hg38/source/onehot/mRNA_pos_41', encode)
# filePath = 'hg38/mRNA/zhong/mRNA_neg_41.txt'
# encode = np.array(onehot(filePath))
# np.save('hg38/source/onehot/mRNA_neg_41', encode)




# # ncpf编码
# filePath = 'hg38/kfold/train/fen/'
# files = os.listdir(filePath)
# for i in files:
#     encode = np.array(ncpf(filePath + i))
#     np.save('hg38/kfold/train/ncpf/' + i[:-4], encode)
# 
# filePath = 'hg38/data/indepent/neg_test_41.txt'
# encode = np.array(ncpf(filePath))
# np.save('hg38/kfold/indepent/ncpf/neg_test_41', encode)
# filePath = 'hg38/data/indepent/pos_test_41.txt'
# encode = np.array(ncpf(filePath))
# np.save('hg38/kfold/indepent/ncpf/pos_test_41', encode)
# 
# filePath = 'hg38/mRNA/zhong/mRNA_pos_41.txt'
# encode = np.array(ncpf(filePath))
# np.save('hg38/source/ncpf/mRNA_pos_41', encode)
# filePath = 'hg38/mRNA/zhong/mRNA_neg_41.txt'
# encode = np.array(ncpf(filePath))
# np.save('hg38/source/ncpf/mRNA_neg_41', encode)




# # # pskp编码
# filePath = 'hg38/kfold/train/fen/'
# files = os.listdir(filePath)
# for i in files:
#     encode = np.array(pskp(filePath + i))
#     np.save('hg38/kfold/train/pskp/' + i[:-4], encode)
# 
# filePath = 'hg38/data/indepent/neg_test_41.txt'
# encode = np.array(pskp(filePath))
# np.save('hg38/kfold/indepent/pskp/neg_test_41', encode)
# filePath = 'hg38/data/indepent/pos_test_41.txt'
# encode = np.array(pskp(filePath))
# np.save('hg38/kfold/indepent/pskp/pos_test_41', encode)
# 
# filePath = 'hg38/mRNA/zhong/mRNA_pos_41.txt'
# encode = np.array(pskp(filePath))
# np.save('hg38/source/pskp/mRNA_pos_41', encode)
# filePath = 'hg38/mRNA/zhong/mRNA_neg_41.txt'
# encode = np.array(pskp(filePath))
# np.save('hg38/source/pskp/mRNA_neg_41', encode)


# # # # PCP编码
# filePath = 'hg38/kfold/train/fen/'
# files = os.listdir(filePath)
# for i in files:
#     encode = np.array(PCP(filePath + i))
#     np.save('hg38/kfold/train/PCP/' + i[:-4], encode)
#
# filePath = 'hg38/data/indepent/neg_test_41.txt'
# encode = np.array(PCP(filePath))
# np.save('hg38/kfold/indepent/PCP/neg_test_41', encode)
# filePath = 'hg38/data/indepent/pos_test_41.txt'
# encode = np.array(PCP(filePath))
# np.save('hg38/kfold/indepent/PCP/pos_test_41', encode)
#
# filePath = 'hg38/mRNA/zhong/mRNA_pos_41.txt'
# encode = np.array(PCP(filePath))
# np.save('hg38/source/PCP/mRNA_pos_41', encode)
# filePath = 'hg38/mRNA/zhong/mRNA_neg_41.txt'
# encode = np.array(PCP(filePath))
# np.save('hg38/source/PCP/mRNA_neg_41', encode)







