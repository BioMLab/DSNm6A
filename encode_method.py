import numpy as np
import math
import pandas as pd

# 打开文件进行独热编码[n,41,4]
def onehot(x):
    f = open(x, 'r')
    ch = []
    squences = f.readlines()
    for i in squences:
        if '>' in i:
            continue
        else:
            chr = []
            for j in i.strip():
                if j == 'N':
                    break
                if j == 'A':
                    chr.append(int(1))
                if j == 'T':
                    chr.append(int(2))
                if j == 'C':
                    chr.append(int(3))
                if j == 'G':
                    chr.append(int(4))
            ch.append(chr)
    f.close()
    # print(11, ch)
    array = np.asarray(ch)
    # print(22, array)
    onehot = np.eye(4)[array - 1]
    # print(33, onehot)
    return onehot


# 理化性质和频率编码，环，氢键，官能团
# [n,41,4]
def ncpf(x):
    f = open(x, 'r')
    ch = []
    sequences = f.readlines()
    for i in sequences:
        if '>' in i:
            continue
        else:
            chr = []
            sum = 0
            shu = []
            for j in i.strip():
                if j == 'A':
                    shu.append(j)
                    sum = sum + 1
                    pin = shu.count(j) / sum
                    chr.append([1, 1, 1, pin])
                if j == 'T':
                    shu.append(j)
                    sum = sum + 1
                    pin = shu.count(j) / sum
                    chr.append([0, 0, 1, pin])
                if j == 'C':
                    shu.append(j)
                    sum = sum + 1
                    pin = shu.count(j) / sum
                    chr.append([0, 1, 0, pin])
                if j == 'G':
                    shu.append(j)
                    sum = sum + 1
                    pin = shu.count(j) / sum
                    chr.append([1, 0, 0, pin])
            ch.append(chr)
    f.close()
    return np.asarray(ch)


nucleotides = ['A', 'C', 'G', 'T']
# 每个位置出现ACGU的概率
def onepin(seqs):
    dict_one = {i: [0]* 141 for i in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql)):
            dict_one[seql[i]][i] += 1
    for i in dict_one.keys():
        for j in range(len(dict_one[i])):
            dict_one[i][j] = dict_one[i][j] / len(seqs)
    return dict_one
def twopin(seqs):
    dict_two = {i + j: [0]* 140 for i in nucleotides for j in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql)-1):
            dict_two[seql[i] + seql[i + 1]][i] += 1
    for i in dict_two.keys():
        for j in range(len(dict_two[i])):
            dict_two[i][j] = dict_two[i][j] / len(seqs)
    return dict_two

def threepin(seqs):
    dict_three = {i + j + k: [0]* 139 for i in nucleotides for j in nucleotides for k in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql) - 2):
            dict_three[seql[i] + seql[i + 1] + seql[i + 2]][i] += 1
    for i in dict_three.keys():
        for j in range(len(dict_three[i])):
            dict_three[i][j] = dict_three[i][j] / len(seqs)
    # print(dict_three)
    return dict_three
# 每个位置出现ACGU的正样本概率减去负样本概率
def pskp_train():
    # pos = 'dataset/DS_H/lncRNA/pos_train.txt'
    # neg = 'dataset/DS_H/lncRNA/neg_train.txt'
    pos = 'dataset/DS_H/mRNA/pos.txt'
    neg = 'dataset/DS_H/mRNA/neg.txt'


    data_pos = open(pos, 'r')
    seqs_pos = data_pos.readlines()
    data_neg = open(neg, 'r')
    seqs_neg = data_neg.readlines()




    onepin_pos = {keys: values for keys, values in onepin(seqs_pos).items()}
    onepin_neg = {keys: values for keys, values in onepin(seqs_neg).items()}
    twopin_pos = {keys: values for keys, values in twopin(seqs_pos).items()}
    twopin_neg = {keys: values for keys, values in twopin(seqs_neg).items()}
    threepin_pos = {keys: values for keys, values in threepin(seqs_pos).items()}
    threepin_neg = {keys: values for keys, values in threepin(seqs_neg).items()}
    ps1p = onepin_pos
    ps2p = twopin_pos
    ps3p = threepin_pos
    for keys, values in ps1p.items():
        for i in range(len(ps1p[keys])):
            ps1p[keys][i] = ps1p[keys][i] - onepin_neg[keys][i]
    for keys, values in ps2p.items():
        for i in range(len(ps2p[keys])):
            ps2p[keys][i] = ps2p[keys][i] - twopin_neg[keys][i]
    for keys, values in ps3p.items():
        for i in range(len(ps3p[keys])):
            ps3p[keys][i] = ps3p[keys][i] - threepin_neg[keys][i]
    data_pos.close()
    data_neg.close()
    return ps1p, ps2p, ps3p
ps1p, ps2p, ps3p = pskp_train()
# print(ps1p)
# [n,120]
def pskp(x):
    f = open(x, 'r')
    data = f.readlines()
    psp = []
    for seq in data:
        psp1 = []
        psp2 = []
        psp3 = []
        seql = seq.strip()
        # print(seql)
        for i in range(len(seql)):
            psp1.append(ps1p[seql[i]][i])
        # print(len(psp1))
        for i in range(len(seql) - 1):
            psp2.append(ps2p[seql[i] + seql[i + 1]][i])
        # print(ps3p)
        # print(len(psp2))
        for i in range(len(seql) - 2):
            psp3.append(ps3p[seql[i] + seql[i + 1] + seql[i + 2]][i])
        # print(len(psp3))
        psp.append(psp1 + psp2 + psp3)
    return np.asarray(psp)


