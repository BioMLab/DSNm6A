import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from model import DSN
from torchmetrics.functional import average_precision, accuracy, auroc, f1_score, specificity, matthews_corrcoef, recall
import numpy as np
from math import sqrt
from sklearn import metrics
from sklearn.metrics import precision_recall_curve

def test(epoch, dataset, fold):

    ###################
    # params          #
    ###################
    cuda = 1
    cudnn.benchmark = False
    ###################
    # load data       #
    ###################

    dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=len(dataset),
            shuffle=True
        )

    ####################
    # load model       #
    ####################

    my_net = DSN()
    checkpoint = torch.load(os.path.join('model/fold' + str(fold) + 'net' + str(epoch) + '.pth'))
    my_net.load_state_dict(checkpoint)
    my_net.eval()

    if cuda:
        my_net = my_net.cuda()

    ####################
    # transform image  #
    ####################

    data_iter = iter(dataloader)
    data_input = next(data_iter)
    pskp_PCP, label = data_input

    batch_size = len(label)

    input_pskp_PCP = torch.FloatTensor(batch_size, 1, 1108)
    class_label = torch.LongTensor(batch_size)

    if cuda:
        pskp_PCP = pskp_PCP.cuda()
        label = label.cuda()
        input_pskp_PCP = input_pskp_PCP.cuda()
        class_label = class_label.cuda()

    input_pskp_PCP.resize_as_(input_pskp_PCP).copy_(pskp_PCP)
    class_label.resize_as_(label).copy_(label)
    inputv_pskp_PCP = Variable(input_pskp_PCP)
    classv_label = Variable(class_label)

    result = my_net(input_data_pskp_PCP=inputv_pskp_PCP, mode='target', rec_scheme='share')

    tp = 0
    fn = 0
    fp = 0
    tn = 0
    thresholds = 0.58
    for z, x in zip(label, result[3]):
        if z > thresholds and x > thresholds:
            tp = tp + 1
        if z > thresholds and x <= thresholds:
            fn = fn + 1
        if z <= thresholds and x > thresholds:
            fp = fp + 1
        if z <= thresholds and x <= thresholds:
            tn = tn + 1
    total = tn + fp + fn + tp
    print('tn: ', tn, ' fp: ', fp, ' fn: ', fn, ' tp: ', tp)
    print('tn: ', tn/total, ' fp: ', fp/total, ' fn: ', fn/total, ' tp: ', tp/total)
    
    sn = np.around(tp / (tp + fn), 6) # recall
    sp = np.around(tn / (tn + fp), 6)
    print(type(result[3]))
    print(result[3].shape)
    print(label.shape)
    print(type(label))
    accu = accuracy(result[3], label.int(), threshold=0.58)
    auc = auroc(result[3], label.int())
    precision = average_precision(result[3], label, pos_label=1)
    f1 = f1_score(result[3], label.int(), threshold=0.58)
    spec = specificity(result[3], label.int(), threshold=0.58)
    mcc = matthews_corrcoef(result[3], label.int(), num_classes=2, threshold=0.58)
    call = recall(result[3], label.int(), threshold=0.58)
    precisions1, recalls1, thresholds1 = precision_recall_curve(label.int().data.cpu().numpy(), result[3].data.cpu().numpy())
    pr = metrics.auc(recalls1, precisions1)

    print('Test: epoch: %d,sn: %f, accuracy: %f, AUC: %f, pre: %f, f1: %f, spec: %f, mcc: %f, recall: %f, sp:%f, pr:%f'
          % (epoch, sn, accu, auc, precision, f1, spec, mcc, call, sp, pr))
    return sn, accu, auc, precision, f1, spec, mcc, call, sp, label, result[3], tn, fp, fn, tp, pr




