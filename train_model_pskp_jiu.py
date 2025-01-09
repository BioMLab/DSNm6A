import torch.utils.data
import os
import shutil
from sklearn.model_selection import KFold
from model_pskp_jiu import DSN, Params
from function import SIMSE, DiffLoss, MSE
from test_pskp_jiu import test
import torch.nn as nn
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

alpha_weight = 0.02
beta_weight = 0.05
gamma_weight = 0.25
class_weight = 1
momentum = 0.9


#######################
# load data         #
#######################

source_pos_onehot = np.load('hg38-101-141/source/onehot/101nt/pos.npy').reshape(-1, 404)
source_neg_onehot = np.load('hg38-101-141/source/onehot/101nt/neg.npy').reshape(-1, 404)
source_pos_ncpf = np.load('hg38-101-141/source/ncpf/101nt/pos.npy').reshape(-1, 404)
source_neg_ncpf = np.load('hg38-101-141/source/ncpf/101nt/neg.npy').reshape(-1, 404)
source_pos_pskp = np.load('hg38-101-141/source/pskp/101nt/pos.npy').reshape(-1, 300)
source_neg_pskp = np.load('hg38-101-141/source/pskp/101nt/neg.npy').reshape(-1, 300)

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
ACC = 0
AUC = 0
PRE = 0
mm = 0
print("this is the fold %d" % (z))
#####################
#  load model       #
#####################
# my_alpha_weight = nn.Parameter(torch.rand(1)*0, requires_grad=True)
# my_beta_weight = nn.Parameter(torch.rand(1) * 0, requires_grad=True)
# my_gamma_weight = nn.Parameter(torch.rand(1)*0, requires_grad=True)

my_alpha_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)
my_beta_weight = nn.Parameter(nn.Parameter(torch.rand(1)*0.5), requires_grad=True)
my_gamma_weight = nn.Parameter(nn.Parameter(torch.rand(1)), requires_grad=True)





my_net = DSN()
my_params = Params()

def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step,
                     step_decay_weight=step_decay_weight):
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr
    return optimizer

optimizer = optim.SGD(my_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer1 = optim.SGD(my_params.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.BCELoss()
loss_recon1 = MSE()
loss_recon2 = SIMSE()
loss_diff = DiffLoss()
loss_similarity = SIMSE()
# loss_similarity = torch.nn.CrossEntropyLoss()

if cuda:
    my_net = my_net.cuda()
    my_params = my_params.cuda()
    my_alpha_weight = my_alpha_weight.cuda()
    my_beta_weight = my_beta_weight.cuda()
    my_gamma_weight = my_gamma_weight.cuda()

    loss_classification = loss_classification.cuda()
    loss_recon1 = loss_recon1.cuda()
    loss_recon2 = loss_recon2.cuda()
    loss_diff = loss_diff.cuda()
    loss_similarity = loss_similarity.cuda()

for p in my_net.parameters():
    p.requires_grad = True
for p in my_params.parameters():
    p.requires_grad = True

#######################
# results           #
#######################
max_target_acc = 0
cor_epoch = 0


train_file_pos_onehot = 'hg38-101-141/kfold/train/onehot/101nt/pos_101_train_' + str(z) + '.npy'
train_file_neg_onehot = 'hg38-101-141/kfold/train/onehot/101nt/neg_101_train_' + str(z) + '.npy'
test_file_pos_onehot = 'hg38-101-141/kfold/train/onehot/101nt/pos_101_test_' + str(z) + '.npy'
test_file_neg_onehot = 'hg38-101-141/kfold/train/onehot/101nt/neg_101_test_' + str(z) + '.npy'
train_file_pos_ncpf = 'hg38-101-141/kfold/train/ncpf/101nt/pos_101_train_' + str(z) + '.npy'
train_file_neg_ncpf = 'hg38-101-141/kfold/train/ncpf/101nt/neg_101_train_' + str(z) + '.npy'
test_file_pos_ncpf = 'hg38-101-141/kfold/train/ncpf/101nt/pos_101_test_' + str(z) + '.npy'
test_file_neg_ncpf = 'hg38-101-141/kfold/train/ncpf/101nt/neg_101_test_' + str(z) + '.npy'
train_file_pos_pskp = 'hg38-101-141/kfold/train/pskp/101nt/pos_101_train_' + str(z) + '.npy'
train_file_neg_pskp = 'hg38-101-141/kfold/train/pskp/101nt/neg_101_train_' + str(z) + '.npy'
test_file_pos_pskp = 'hg38-101-141/kfold/train/pskp/101nt/pos_101_test_' + str(z) + '.npy'
test_file_neg_pskp = 'hg38-101-141/kfold/train/pskp/101nt/neg_101_test_' + str(z) + '.npy'

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



fxn = open('xneng/xneng_test.txt', 'w')
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
        my_params.zero_grad()

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

        params_my = my_params()


        target_privte_code_pskp_PCP, target_share_code_pskp_PCP, \
            target_domain_label, target_class_label, target_rec_code_pskp_PCP = result

        # print(params_my)
        # print(type(params_my))
        gamma_weight = params_my[2]
        target_dann = loss_similarity(target_domain_label.to(torch.float), torch.reshape(target_domainv_label, (-1, 1)).to(torch.float))
        loss = loss + gamma_weight * target_dann

        class_weight = class_weight
        target_classification = loss_classification(target_class_label.to(torch.float), target_classv_label.to(torch.float))
        loss = loss + class_weight * target_classification


        beta_weight = params_my[1]
        target_diff_pskp_PCP = loss_diff(target_privte_code_pskp_PCP, target_share_code_pskp_PCP)
        loss = loss + beta_weight * target_diff_pskp_PCP


        alpha_weight = params_my[0]
        target_mse_pskp_PCP = loss_recon1(target_rec_code_pskp_PCP, target_inputv_pskp_PCP)
        loss = loss + alpha_weight * target_mse_pskp_PCP
        target_simse_pskp_PCP = loss_recon2(target_rec_code_pskp_PCP, target_inputv_pskp_PCP)
        loss = loss + alpha_weight * target_simse_pskp_PCP

        loss.backward()
        optimizer.step()

        # my_params.zero_grad()
        optimizer1.step()

        ###################################
        # source data training            #
        ###################################
        for sour_i in range(5):
            data_source = next(data_source_iter)
            s_pskp_PCP, s_label = data_source
        
        
            my_net.zero_grad()
            my_params.zero_grad()
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

            params_my = my_params()


            source_privte_code_pskp_PCP, source_share_code_pskp_PCP, \
                source_domain_label, source_class_label, source_rec_code_pskp_PCP = result


            gamma_weight = params_my[2]
            source_dann = loss_similarity(source_domain_label, torch.reshape(source_domainv_label, (-1, 1)).to(torch.float))
            loss = loss + gamma_weight * source_dann

            class_weight = class_weight
            source_classification = loss_classification(source_class_label.to(torch.float), source_classv_label.to(torch.float))
            loss = loss + class_weight * source_classification

            beta_weight = params_my[1]
            source_diff_pskp_PCP = loss_diff(source_privte_code_pskp_PCP, source_share_code_pskp_PCP)
            loss = loss + beta_weight * source_diff_pskp_PCP

            alpha_weight = params_my[0]
            source_mse_pskp_PCP = loss_recon1(source_rec_code_pskp_PCP, source_inputv_pskp_PCP)
            loss = loss + alpha_weight * source_mse_pskp_PCP
            source_simse_pskp_PCP = loss_recon2(source_rec_code_pskp_PCP, source_inputv_pskp_PCP)
            loss = loss + alpha_weight * source_simse_pskp_PCP
        
            loss.backward()
            optimizer = exp_lr_scheduler(optimizer=optimizer, step=current_step)
            optimizer.step()

            # my_params.zero_grad()
            optimizer1.step()

        i = i + 1
        current_step = current_step + 1


    print(
        'Train specific loss: target_classification: %f,source_classification: %f, \n '
        'Train specific loss: class_weight * target_classification: %f, class_weight * source_classification: %f, \n '
        
        
        'source_dann: %f, source_diff_pskp_PCP: %f, ' \
        'gamma_weight * source_dann: %f, beta_weight * source_diff_pskp_PCP: %f, ' \
        
        
        'source_mse_pskp_PCP: %f, source_simse_pskp_PCP: %f, \n target_dann: %f,'
        'alpha_weight * source_mse_pskp_PCP: %f, alpha_weight * source_simse_pskp_PCP: %f, \n gamma_weight * target_dann: %f,'
        
        
        
        ', target_diff_pskp_PCP: %f, ' \
        ', beta_weight * target_diff_pskp_PCP: %f, ' \
        
        
        'target_mse_pskp_PCP: %f, target_simse_pskp_PCP: %f' \
        'target_mse_pskp_PCP: %f, target_simse_pskp_PCP: %f' \
        % (target_classification.data.cpu().numpy(), source_classification.data.cpu().numpy(),
           class_weight * target_classification.data.cpu().numpy(), class_weight * source_classification.data.cpu().numpy(),

           source_dann.data.cpu().numpy(),  # Tensor.numpy()将Tensor转化为ndarray
           gamma_weight * source_dann,  # Tensor.numpy()将Tensor转化为ndarray
           # .cpu将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
           # 返回和 x 的相同数据 tensor, 而且这个新的tensor和原来的tensor是共用数据的,一者改变，另一者也会跟着改变



           source_diff_pskp_PCP.data.cpu().numpy(),
           beta_weight * source_diff_pskp_PCP,



           source_mse_pskp_PCP.data.cpu().numpy(),
           alpha_weight * source_mse_pskp_PCP,
           source_simse_pskp_PCP.data.cpu().numpy(),
           alpha_weight * source_simse_pskp_PCP,
           target_dann.data.cpu().numpy(),
           gamma_weight * target_dann,



           target_diff_pskp_PCP.data.cpu().numpy(),
           beta_weight * target_diff_pskp_PCP,


           target_mse_pskp_PCP.data.cpu().numpy(),
           target_simse_pskp_PCP.data.cpu().numpy(),
           alpha_weight * target_mse_pskp_PCP,
           alpha_weight * target_simse_pskp_PCP))
    print('step: %d, loss: %f' % (current_step, loss.cpu().data.numpy()))

    torch.save(my_net.state_dict(), 'model/fold' + str(z) + 'net' + str(epoch) + '.pth')
    sn, accu, auc, precision, f1, spec, mcc, call, sp, labe, resul, tn, fp, fn, tp, pr = test(epoch=epoch, dataset=dataset_target_test, fold = z)
    if epoch >= 1:
        mm = mm + 1
        ACC = ACC + accu
        AUC = AUC + auc
        PRE = PRE + precision
        print('AVG ACC:', ACC/mm)
        print('AVG AUC:', AUC/mm)
        print('AVG PRE:', PRE/mm)
    # if auc > max_target_acc:
        max_target_acc = auc
        cor_epoch = epoch

        fxn.write('***************************第' + str(epoch) + '轮的训练结果*********************************\n')
        fxn.write('***************************第' + str(epoch) + '轮的训练结果*********************************\n')
        fxn.write('xneng/xneng_test:\n')
        total = tn + fp + fn + tp
        fxn.write('tn: ' + str(tn) + ' fp: ' + str(fp) + ' fn: ' + str(fn) + ' tp: ' + str(tp) + '\n')
        fxn.write('tn: ' + str(tn/total) + ' fp: ' + str(fp/total) + ' fn: ' + str(fn/total) + ' tp: ' + str(tp/total) + '\n')
        fxn.write('Test: epoch: %d,\nsn: %f, accuracy: %f, AUC: %f, pre: %f, f1: %f, spec: %f, mcc: %f, recall: %f, sp:%f, pr:%f\n'
          % (epoch, sn, accu, auc, precision, f1, spec, mcc, call, sp, pr))
        fxn.write("第%d轮得到的alpha_weight的参数值:%f\n"% (epoch, alpha_weight))
        fxn.write("第%d轮得到的beta_weight的参数值:%f\n"% (epoch, beta_weight))
        fxn.write("第%d轮得到的gamma_weight的参数值:%f\n"% (epoch, gamma_weight))
        fxn.write("第%d轮得到的class_weight的参数值:%f\n"% (epoch, class_weight))


        fxn.write(
            '---------------------------------------------------------------------------------\n'
            'Train specific loss: \ntarget_classification: %f,\n'
            'target_dann: %f,\n'
            'target_diff_pskp_PCP: %f,\n'
            'target_mse_pskp_PCP: %f, \n'
            'target_simse_pskp_PCP: %f\n'

            'Train specific loss: \nclass_weight * target_classification: %f,\n'
            'gamma_weight * target_dann: %f,\n'
            'beta_weight * target_diff_pskp_PCP: %f,\n'
            'alpha_weight * target_mse_pskp_PCP: %f,\n'
            'alpha_weight * target_simse_pskp_PCP: %f\n'

            '---------------------------------------------------------------------------------\n'
            'Train specific loss: \nsource_classification: %f, \n '
            'source_dann: %f, '
            'source_diff_pskp_PCP: %f,'
            'source_mse_pskp_PCP: %f, '
            'source_simse_pskp_PCP: %f,'

            'Train specific loss: \nclass_weight * source_classification: %f, \n'
            'gamma_weight * source_dann: %f, \n'
            'beta_weight * source_diff_pskp_PCP: %f, \n'
            'alpha_weight * source_mse_pskp_PCP: %f, \n'
            'alpha_weight * source_simse_pskp_PCP: %f,\n'
            '---------------------------------------------------------------------------------\n'


            % (target_classification.data.cpu().numpy(),
               target_dann.data.cpu().numpy(),
               target_diff_pskp_PCP.data.cpu().numpy(),
               target_mse_pskp_PCP.data.cpu().numpy(),
               target_simse_pskp_PCP.data.cpu().numpy(),

               class_weight * target_classification.data.cpu().numpy(),
               gamma_weight * target_dann,
               beta_weight * target_diff_pskp_PCP,
               alpha_weight * target_mse_pskp_PCP,
               alpha_weight * target_simse_pskp_PCP,


               source_classification.data.cpu().numpy(),
               source_dann.data.cpu().numpy(),  # Tensor.numpy()将Tensor转化为ndarray
               source_diff_pskp_PCP.data.cpu().numpy(),
               source_mse_pskp_PCP.data.cpu().numpy(),
               source_simse_pskp_PCP.data.cpu().numpy(),

               class_weight * source_classification.data.cpu().numpy(),
               gamma_weight * source_dann,  # Tensor.numpy()将Tensor转化为ndarray
               # .cpu将数据的处理设备从其他设备（如.cuda()拿到cpu上），不会改变变量类型，转换后仍然是Tensor变量。
               # 返回和 x 的相同数据 tensor, 而且这个新的tensor和原来的tensor是共用数据的,一者改变，另一者也会跟着改变
               beta_weight * source_diff_pskp_PCP,
               alpha_weight * source_mse_pskp_PCP,
               alpha_weight * source_simse_pskp_PCP
               )
        )

        fxn.write('step: %d, loss: %f\n' % (current_step, loss.cpu().data.numpy()))




        fjie = open('xneng/jie_test' + str(epoch) + '.txt', 'w')
        fjie.write('xneng/jie_test:')
        for i in range(len(labe)):
            fjie.write(str(labe[i]) + '\t' + str(resul[i]) + '\n')
        fjie.close()

        print("第%d轮得到的alpha_weight的参数值:%f"% (epoch, alpha_weight))
        print("第%d轮得到的beta_weight的参数值:%f"% (epoch, beta_weight))
        print("第%d轮得到的gamma_weight的参数值:%f"% (epoch, gamma_weight))
        print("第%d轮得到的class_weight的参数值:%f"% (epoch, class_weight))

    print('Current maximum acc: %f, epoch: %d' % (max_target_acc, cor_epoch) )
    print("-------------------------------------------------------------------------------------")

fxn.write("=============================================================")
fxn.write("=============================================================")
fxn.write("===========================最终结果==================================")
fxn.write("=============================================================")
fxn.write("=============================================================")
fxn.write("最终得到的alpha_weight的参数值:%f", alpha_weight)
fxn.write("最终得到的beta_weight的参数值:%f", beta_weight)
fxn.write("最终得到的gamma_weight的参数值:%f", gamma_weight)
fxn.write("最终得到的class_weight的参数值:%f", class_weight)
fxn.write('done')
fxn.close()
print("最终得到的alpha_weight的参数值:%f", alpha_weight)
print("最终得到的beta_weight的参数值:%f", beta_weight)
print("最终得到的gamma_weight的参数值:%f", gamma_weight)
print("最终得到的class_weight的参数值:%f", class_weight)
print('done')









# RuntimeError: CUDA error: device-side assert triggered
# CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
#
#
# RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED when calling `cublasSgemm( handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc)`






