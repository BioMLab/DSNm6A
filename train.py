import torch.utils.data
import os
import shutil
from sklearn.model_selection import KFold
from model import DSN
from function import SIMSE, DiffLoss, MSE
from test import test

import random
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch.autograd import Variable

seed = 2
print(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


######################
# params             #
######################

model_root = 'model'
cuda = 1
cudnn.benchmark = True
lr = 0.005
batch_size = 32
n_epoch = 25
step_decay_weight = 0.9
lr_decay_step = 600
active_domain_loss_step = 100000
weight_decay = 1e-5

momentum = 0.9

if not os.path.exists('model'):
    # 若不存在则创建文件夹
    print(111)
    os.makedirs('model')
else:
    print(22222)

if not os.path.exists('xneng'):
    # 若不存在则创建文件夹
    print(333333)
    os.makedirs('xneng')
else:
    print(444444)
#######################
# load data         #
#######################

source_pos_onehot = np.load('encode/DS_H/mRNA/onehot/pos.npy').reshape(-1, 404)
source_neg_onehot = np.load('encode/DS_H/mRNA/onehot/neg.npy').reshape(-1, 404)
source_pos_ncpf = np.load('encode/DS_H/mRNA/ncpf/pos.npy').reshape(-1, 404)
source_neg_ncpf = np.load('encode/DS_H/mRNA/ncpf/neg.npy').reshape(-1, 404)
source_pos_pskp = np.load('encode/DS_H/mRNA/pskp/pos.npy').reshape(-1, 300)
source_neg_pskp = np.load('encode/DS_H/mRNA/pskp/neg.npy').reshape(-1, 300)

source_pos_onehot_ncpf = np.concatenate((source_pos_onehot, source_pos_ncpf, source_pos_pskp), axis=1)
source_neg_onehot_ncpf = np.concatenate((source_neg_onehot, source_neg_ncpf, source_neg_pskp), axis=1)

source_train_onehot_ncpf = np.vstack((source_pos_onehot_ncpf,
                                      source_neg_onehot_ncpf))

source_train_label = torch.cat(
    (torch.ones(source_pos_onehot.shape[0], 1), torch.zeros(source_neg_onehot.shape[0], 1))
    , dim=0
)

source_train_onehot_ncpf = torch.tensor(np.expand_dims(source_train_onehot_ncpf, axis=1), dtype=torch.float32).type(torch.FloatTensor)


dataset_source = torch.utils.data.TensorDataset(source_train_onehot_ncpf, source_train_label)


dataloader_source = torch.utils.data.DataLoader(
    dataset=dataset_source,
    batch_size=batch_size,
    shuffle=True
)


z = 1

print("this is the fold %d" % (z))
#####################
#  load model       #
#####################

my_net = DSN()

def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step,
                     step_decay_weight=step_decay_weight):
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return optimizer

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.BCELoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True

#######################
# results           #
#######################
max_target_acc = 0
cor_epoch = 0


train_file_pos_onehot = 'encode/DS_H/lncRNA/onehot/pos_train.npy'
train_file_neg_onehot = 'encode/DS_H/lncRNA/onehot/neg_train.npy'
test_file_pos_onehot = 'encode/DS_H/lncRNA/onehot/pos_test.npy'
test_file_neg_onehot = 'encode/DS_H/lncRNA/onehot/neg_test.npy'
train_file_pos_ncpf = 'encode/DS_H/lncRNA/ncpf/pos_train.npy'
train_file_neg_ncpf = 'encode/DS_H/lncRNA/ncpf/neg_train.npy'
test_file_pos_ncpf = 'encode/DS_H/lncRNA/ncpf/pos_test.npy'
test_file_neg_ncpf = 'encode/DS_H/lncRNA/ncpf/neg_test.npy'
train_file_pos_pskp = 'encode/DS_H/lncRNA/pskp/pos_train.npy'
train_file_neg_pskp = 'encode/DS_H/lncRNA/pskp/neg_train.npy'
test_file_pos_pskp = 'encode/DS_H/lncRNA/pskp/pos_test.npy'
test_file_neg_pskp = 'encode/DS_H/lncRNA/pskp/neg_test.npy'

#
train_data_pos_onehot = np.load(train_file_pos_onehot).reshape(-1, 404)
train_data_neg_onehot = np.load(train_file_neg_onehot).reshape(-1, 404)
test_data_pos_onehot = np.load(test_file_pos_onehot).reshape(-1, 404)
test_data_neg_onehot = np.load(test_file_neg_onehot).reshape(-1, 404)
train_data_pos_ncpf = np.load(train_file_pos_ncpf).reshape(-1, 404)
train_data_neg_ncpf = np.load(train_file_neg_ncpf).reshape(-1, 404)
test_data_pos_ncpf = np.load(test_file_pos_ncpf).reshape(-1, 404)
test_data_neg_ncpf = np.load(test_file_neg_ncpf).reshape(-1, 404)
train_data_pos_pskp = np.load(train_file_pos_pskp).reshape(-1, 300)
train_data_neg_pskp = np.load(train_file_neg_pskp).reshape(-1, 300)
test_data_pos_pskp = np.load(test_file_pos_pskp).reshape(-1, 300)
test_data_neg_pskp = np.load(test_file_neg_pskp).reshape(-1, 300)

train_pos_onehot_ncpf = np.concatenate((train_data_pos_onehot, train_data_pos_ncpf, train_data_pos_pskp), axis=1)
train_neg_onehot_ncpf = np.concatenate((train_data_neg_onehot, train_data_neg_ncpf, train_data_neg_pskp), axis=1)
test_pos_onehot_ncpf = np.concatenate((test_data_pos_onehot, test_data_pos_ncpf, test_data_pos_pskp), axis=1)
test_neg_onehot_ncpf = np.concatenate((test_data_neg_onehot, test_data_neg_ncpf, test_data_neg_pskp), axis=1)

target_train_onehot_ncpf = np.vstack((train_pos_onehot_ncpf,
                                      train_neg_onehot_ncpf))
target_test_onehot_ncpf = np.vstack((test_pos_onehot_ncpf,
                                     test_neg_onehot_ncpf))


target_train_label = torch.cat(
    (torch.ones(train_data_pos_onehot.shape[0], 1),
     torch.zeros(train_data_neg_onehot.shape[0], 1))
    , dim=0
)

target_test_label = torch.cat(
    (torch.ones(test_data_pos_onehot.shape[0], 1),
     torch.zeros(test_data_neg_onehot.shape[0], 1))
    , dim=0
)

target_train_onehot_ncpf = torch.tensor(np.expand_dims(target_train_onehot_ncpf, axis=1), dtype=torch.float32).type(torch.FloatTensor)
target_test_onehot_ncpf = torch.tensor(np.expand_dims(target_test_onehot_ncpf, axis=1), dtype=torch.float32).type(torch.FloatTensor)

dataset_target = torch.utils.data.TensorDataset(target_train_onehot_ncpf, target_train_label)
dataset_target_test = torch.utils.data.TensorDataset(target_test_onehot_ncpf, target_test_label)

dataloader_target = torch.utils.data.DataLoader(
    dataset=dataset_target,
    batch_size=batch_size,
    shuffle=True
)

len_dataloader = min(len(dataloader_source), len(dataloader_target))
dann_epoch = np.floor(active_domain_loss_step / len_dataloader * 1.0)


# ###########################
# training network          #
# ###########################

current_step = 0
fk = open('xneng/y_test_prodict.txt', 'w')
fk.write('y_test_xneng:\n')
for epoch in range(1, n_epoch+1):
    # </editor-fold>
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        ###################################
        # target data training            #
        ###################################

        data_target = next(data_target_iter)
        t_pskp_PCP, t_label = data_target

        my_net.zero_grad()
        loss = 0
        batch_size = len(t_label)

        input_pskp_PCP = torch.FloatTensor(batch_size, 1, 1108)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        t_pskp_PCP = t_pskp_PCP.float()
        t_label = t_label.float()
        input_pskp_PCP = input_pskp_PCP.float()
        class_label = class_label.float()
        domain_label = domain_label.float()

        if cuda:
            t_pskp_PCP = t_pskp_PCP.cuda()
            t_label = t_label.cuda()
            input_pskp_PCP = input_pskp_PCP.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_pskp_PCP.resize_as_(t_pskp_PCP).copy_(t_pskp_PCP)
        class_label.resize_as_(t_label).copy_(t_label)
        target_inputv_pskp_PCP = Variable(input_pskp_PCP)
        target_classv_label = Variable(class_label)
        target_domainv_label = Variable(domain_label)

        p = float(i + (epoch - dann_epoch) * len_dataloader / (n_epoch - dann_epoch) / len_dataloader)
        p = 2. / (1. + np.exp(-10 * p)) - 1

        # activate domain loss
        result = my_net(input_data_pskp_PCP=target_inputv_pskp_PCP,
                        mode='target', rec_scheme='all', p=p)

        target_privte_code_pskp_PCP, target_share_code_pskp_PCP, \
            target_domain_label, target_class_label, target_rec_code_pskp_PCP = result

        target_dann = my_net.gamma_weight * loss_similarity(target_domain_label.to(torch.float), torch.reshape(target_domainv_label, (-1, 1)).to(torch.float))
        loss = loss + target_dann

        target_classification = loss_classification(target_class_label.to(torch.float), target_classv_label.to(torch.float))
        loss = loss + target_classification

        target_diff_pskp_PCP = my_net.beta_weight * loss_diff(target_privte_code_pskp_PCP, target_share_code_pskp_PCP)
        loss = loss + target_diff_pskp_PCP

        target_mse_pskp_PCP = my_net.alpha_weight * loss_recon1(target_rec_code_pskp_PCP, target_inputv_pskp_PCP)
        loss = loss + target_mse_pskp_PCP
        target_simse_pskp_PCP = my_net.alpha_weight * loss_recon2(target_rec_code_pskp_PCP, target_inputv_pskp_PCP)
        loss = loss + target_simse_pskp_PCP

        my_net.alpha_weight.data = torch.clamp(my_net.alpha_weight, min=0.01, max=0.05)
        my_net.beta_weight.data = torch.clamp(my_net.beta_weight, min=0.01, max=0.1)
        my_net.gamma_weight.data = torch.clamp(my_net.gamma_weight, min=0.1, max=0.5)

        loss.backward()
        optimizer.step()

        ###################################
        # source data training            #
        ###################################
        for sour_i in range(5):
            data_source = next(data_source_iter)
            s_pskp_PCP, s_label = data_source
        
        
            my_net.zero_grad()
            batch_size = len(s_label)
        
            input_pskp_PCP = torch.FloatTensor(batch_size, 1, 1108)
            class_label = torch.LongTensor(batch_size)
            domain_label = torch.zeros(batch_size)
            domain_label = domain_label.long()
        
            loss = 0
        
            if cuda:
                s_pskp_PCP = s_pskp_PCP.cuda()
                s_label = s_label.cuda()
                input_pskp_PCP = input_pskp_PCP.cuda()
                class_label = class_label.cuda()
                domain_label = domain_label.cuda()
        
            s_label = s_label.long()
        
            input_pskp_PCP.resize_as_(input_pskp_PCP).copy_(s_pskp_PCP)
            class_label.resize_as_(s_label).copy_(s_label)
            source_inputv_pskp_PCP = Variable(input_pskp_PCP)
            source_classv_label = Variable(class_label)
            source_domainv_label = Variable(domain_label)
        
            result = my_net(input_data_pskp_PCP=source_inputv_pskp_PCP,
                            mode='source', rec_scheme='all', p=p)
        
            source_privte_code_pskp_PCP, source_share_code_pskp_PCP, \
                source_domain_label, source_class_label, source_rec_code_pskp_PCP = result
        
            source_dann = my_net.gamma_weight * loss_similarity(source_domain_label, torch.reshape(source_domainv_label, (-1, 1)).to(torch.float))
            loss = loss + source_dann
        
            source_classification = loss_classification(source_class_label.to(torch.float), source_classv_label.to(torch.float))
            loss = loss + source_classification
        
            source_diff_pskp_PCP = my_net.beta_weight * loss_diff(source_privte_code_pskp_PCP, source_share_code_pskp_PCP)
            loss = loss + source_diff_pskp_PCP
        
            source_mse_pskp_PCP = my_net.alpha_weight * loss_recon1(source_rec_code_pskp_PCP, source_inputv_pskp_PCP)
            loss = loss + source_mse_pskp_PCP
            source_simse_pskp_PCP = my_net.alpha_weight * loss_recon2(source_rec_code_pskp_PCP, source_inputv_pskp_PCP)
            loss = loss + source_simse_pskp_PCP

            my_net.alpha_weight.data = torch.clamp(my_net.alpha_weight, min=0.01, max=0.05)
            my_net.beta_weight.data = torch.clamp(my_net.beta_weight, min=0.01, max=0.1)
            my_net.gamma_weight.data = torch.clamp(my_net.gamma_weight, min=0.1, max=0.5)

            loss.backward()
            optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
            optimizer.step()

        i = i + 1
        current_step = current_step + 1


    torch.save(my_net.state_dict(), 'model/fold' + str(z) + 'net' + str(epoch) + '.pth')
    sn, accu, auc, precision, f1, spec, mcc, call, sp, labe, resul, tn, fp, fn, tp, pr = test(epoch=epoch, dataset=dataset_target_test, fold = z)

    fk.write('xneng/xneng_test_' + str(epoch) + '.txt\n')
    fk.write('xneng/xneng_test:')
    total = tn + fp + fn + tp
    fk.write('tn: ' + str(tn) + ' fp: ' + str(fp) + ' fn: ' + str(fn) + ' tp: ' + str(tp) + '\n')
    fk.write('tn: ' + str(tn / total) + ' fp: ' + str(fp / total) + ' fn: ' + str(fn / total) + ' tp: ' + str(
        tp / total) + '\n')
    fk.write('Test: epoch: %d,sn: %f, accuracy: %f, AUC: %f, pre: %f, f1: %f, spec: %f, mcc: %f, recall: %f, sp:%f, pr:%f\n\n\n'
              % (epoch, sn, accu, auc, precision, f1, spec, mcc, call, sp, pr))

    fjie = open('xneng/jie_test' + str(epoch) + '.txt', 'w')
    fjie.write('xneng/jie_test:')
    for i in range(len(labe)):
        fjie.write(str(labe[i]) + '\t' + str(resul[i]) + '\n')
    fjie.close()


    if auc > max_target_acc:
        max_target_acc = auc
        cor_epoch = epoch

    print('Current maximum acc: %f, epoch: %d' % (max_target_acc, cor_epoch) )
    print("-------------------------------------------------------------------------------------")

print('done')




