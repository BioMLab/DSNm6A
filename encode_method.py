import numpy as np
import math
import pandas as pd
# DNA->RNA
# f = open('data_primary/lnc/lnc_fulltx_neg.txt', 'r')
# f1 = open('data_primary/lnc/rna_lnc_fulltx_neg.txt', 'w')
# data = f.readlines()
# # print(data)
# for i in data:
#     ch = ''
#     for j in i[5: 46]:
#         if j == 'T':
#             ch = ch + 'U'
#         else:
#             ch = ch + j
#     # print(ch)
#     ch = ch + '\n'
#     f1.write(ch)
# f1.close()
# f.close()
# f = open('data_primary/lnc/rna_lnc_exonic_neg.txt', 'r')
# data = f.readlines()
# ch = []
# for i in data:
#     ch.append(i[20])
# print(set(ch))
# f.close()

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

# source_pos = onehot('data_primary/mrna/rna_pos.txt')
# print(source_pos.shape)
# print(source_pos)
# print(type(source_pos))
# source_neg = onehot('data_primary/mrna/rna_neg.txt')
# target_pos = onehot('data_primary/lnc/rna_lnc_fulltx_pos.txt')
# target_neg = onehot('data_primary/lnc/rna_lnc_fulltx_neg.txt')
#
# print(source_pos.shape)
# print(source_neg.shape)
# print(target_pos.shape)
# print(target_neg.shape)

# 二核苷酸独热编码
twohot_dict = {
    'AA': [0, 0, 0, 0], 'AC': [0, 0, 0, 1], 'AG': [0, 0, 1, 0], 'AU': [0, 0, 1, 1],
    'CA': [0, 1, 0, 0], 'CC': [0, 1, 0, 1], 'CG': [0, 1, 1, 0], 'CU': [0, 1, 1, 1],
    'GA': [1, 0, 0, 0], 'GC': [1, 0, 0, 1], 'GG': [1, 0, 1 ,0], 'GU': [1, 0, 1, 1],
    'UA': [1, 1, 0, 0], 'UC': [1, 1, 0, 1], 'UG': [1, 1, 1, 0], 'UU': [1, 1, 1, 1]
}
# [n,40,4]
def twohot(x):
    f = open(x, 'r')
    sequences = f.readlines()
    ch = []
    for i in sequences:
        chr = []
        for j in range(len(i.strip())-1):
            chr.append(twohot_dict[i[j: j+2]])
        ch.append(chr)
    f.close()
    ch = np.asarray(ch)
    return ch


# source_pos = twohot('data_primary/mrna/rna_pos.txt')
# source_neg = twohot('data_primary/mrna/rna_neg.txt')
# target_pos = twohot('data_primary/lnc/rna_lnc_fulltx_pos.txt')
# target_neg = twohot('data_primary/lnc/rna_lnc_fulltx_neg.txt')
#
# print(source_pos.shape)
# print(source_neg.shape)
# print(target_pos.shape)
# print(target_neg.shape)

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

# source_pos = NCPF('data_primary/mrna/rna_pos.txt')
# source_neg = NCPF('data_primary/mrna/rna_neg.txt')
# target_pos = NCPF('data_primary/lnc/rna_lnc_fulltx_pos.txt')
# target_neg = NCPF('data_primary/lnc/rna_lnc_fulltx_neg.txt')
#
# print(source_pos.shape)
# print(source_neg.shape)
# print(target_pos.shape)
# print(target_neg.shape)




nucleotides = ['A', 'C', 'G', 'T']
# 每个位置出现ACGU的概率
def onepin(seqs):
    dict_one = {i: [0]* 41 for i in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql)):
            dict_one[seql[i]][i] += 1
    for i in dict_one.keys():
        for j in range(len(dict_one[i])):
            dict_one[i][j] = dict_one[i][j] / len(seqs)
    return dict_one
def twopin(seqs):
    dict_two = {i + j: [0]* 40 for i in nucleotides for j in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql)-1):
            dict_two[seql[i] + seql[i + 1]][i] += 1
    for i in dict_two.keys():
        for j in range(len(dict_two[i])):
            dict_two[i][j] = dict_two[i][j] / len(seqs)
    return dict_two

def threepin(seqs):
    dict_three = {i + j + k: [0]* 49 for i in nucleotides for j in nucleotides for k in nucleotides}
    for seq in seqs:
        seql = seq.strip()
        for i in range(len(seql) - 2):
            dict_three[seql[i] + seql[i + 1] + seql[i + 2]][i] += 1
    for i in dict_three.keys():
        for j in range(len(dict_three[i])):
            dict_three[i][j] = dict_three[i][j] / len(seqs)
    return dict_three
# 每个位置出现ACGU的正样本概率减去负样本概率
def pskp_train():
    # f = open('other-12/Arabidopsis Thaliana-TAIR10_CL_Tech/source/mRNA_neg_41.txt')
    pos = '../../../../mRNA_pos_41.txt'
    neg = '../../../../mRNA_neg_41.txt'

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
        for i in range(len(seql)):
            psp1.append(ps1p[seql[i]][i])
        for i in range(len(seql) - 1):
            psp2.append(ps2p[seql[i] + seql[i + 1]][i])
        for i in range(len(seql) - 2):
            psp3.append(ps3p[seql[i] + seql[i + 1] + seql[i + 2]][i])
        psp.append(psp1 + psp2 + psp3)
    return np.asarray(psp)
#
# source_pos = pskp-lncRNA('data_primary/mrna/rna_pos.txt')
# source_neg = pskp-lncRNA('data_primary/mrna/rna_neg.txt')
# target_pos = pskp-lncRNA('data_primary/lnc/rna_lnc_fulltx_pos.txt')
# target_neg = pskp-lncRNA('data_primary/lnc/rna_lnc_fulltx_neg.txt')
#
# print(source_pos.shape)
# print(source_neg.shape)
# print(target_pos.shape)
# print(target_neg.shape)


# [n,400]
# 二核苷酸物理化学性质编码
def PCP(path):
    # path = ""
    gene_type = "RNA"
    fill_NA = '0'
    propertyname = "physical_chemical_properties_DNA.txt"
    physical_chemical_properties_path = propertyname

    fh = open(path, 'r')
    seq = []
    for line in fh:  # get the fasta sequence
        if line.startswith('>'):
            pass
        else:
            seq.append(line.replace('\n', '').replace('\r', ''))
    fh.close()

    data = pd.read_csv(physical_chemical_properties_path, header=None,
                       index_col=None)  # read the phisical chemichy proporties
    # print(data)
    prop_key = data.values[:, 0]
    # print(prop_key)

    if fill_NA == "1":
        prop_key[21] = 'NA'
    # print(prop_key)
    prop_data = data.values[:, 1:]
    prop_data = np.matrix(prop_data)  # 取值16*10
    # print(prop_data)
    DNC_value = np.array(prop_data).T  # 转置 10*16
    # print(DNC_value)
    DNC_value_scale = [[]] * len(DNC_value)  # 10*0
    # print(DNC_value_scale)
    for i in list(range(len(DNC_value))):
        # print(i)
        # print(len(DNC_value[i]))
        average_ = sum(DNC_value[i] * 1.0 / len(DNC_value[i]))  # 16个元素求和取均值
        # print(average_)
        std_ = np.std(DNC_value[i], ddof=1)  # np,std求标准差,当ddo = 1时，表示计算无偏样本标准差，最终除以n-1
        # print(std_)
        DNC_value_scale[i] = [round((e - average_) / std_, 2) for e in
                              DNC_value[i]]  # round() 方法返回浮点数x的四舍五入值,  10*16的元素减去均值/方差
    # print(DNC_value_scale)
    prop_data_transformed = list(zip(*DNC_value_scale))  # 16*10的列表，每个元素是元组
    # print(len(prop_data_transformed))
    # prop_data_transformed=StandardScaler().fit_transform(prop_data)
    prop_len = len(prop_data_transformed[0])  # 就是10

    whole_m6a_seq = seq
    i = 0
    phisical_chemichy_len = len(prop_data_transformed)  # the length of properties 就是16
    sequence_line_len = len(seq[0])  # the length of one.txt sequence 就是41
    LAMDA = 4
    finally_result = []  # used to save the fanal result
    for one_m6a_sequence_line in whole_m6a_seq:
        one_sequence_value = [[]] * (sequence_line_len - 1)  # 列表含有40个空列表元素
        # print(one_sequence_value)
        PC_m = [0.0] * prop_len  # 列表含有10给0.0
        PC_m = np.array(PC_m)  # 换为数组
        # print(PC_m)
        for one_sequence_index in range(sequence_line_len - 1):
            for prop_index in list(range(len(prop_key))):  # 0-15
                # print(prop_index)
                if one_m6a_sequence_line[one_sequence_index:one_sequence_index + 2] == prop_key[prop_index]:
                    one_sequence_value[one_sequence_index] = prop_data_transformed[prop_index]  # 根据序列位置加入相应的10个元素
                    # print(np.array(one_sequence_value[one_sequence_index]))
            PC_m += np.array(one_sequence_value[one_sequence_index])  # 最后10个元素的数组,是每个位置的10个值的加和
        # print(PC_m)
        PC_m = PC_m / (sequence_line_len - 1)
        auto_value = []
        for LAMDA_index in list(range(1, LAMDA + 1)):
            temp = [0.0] * prop_len
            temp = np.array(temp)
            for auto_index in list(range(1, sequence_line_len - LAMDA_index)):
                temp = temp + (np.array(one_sequence_value[auto_index - 1]) - PC_m) * (
                            np.array(one_sequence_value[auto_index + LAMDA_index - 1]) - PC_m)
                temp = [round(e, 8) for e in temp.astype(float)]
            x = [round(e / (sequence_line_len - LAMDA_index - 1), 8) for e in temp]
            auto_value.extend([round(e, 8) for e in x])
        for LAMDA_index in list(range(1, LAMDA + 1)):
            for i in list(range(1, prop_len + 1)):
                for j in list(range(1, prop_len + 1)):
                    temp2 = 0.0
                    if i != j:
                        for auto_index in list(range(1, sequence_line_len - LAMDA_index)):
                            temp2 += (one_sequence_value[auto_index - 1][i - 1] - PC_m[i - 1]) * (
                                        one_sequence_value[auto_index + LAMDA_index - 1][j - 1] - PC_m[j - 1])
                        auto_value.append(round(temp2 / ((sequence_line_len - 1) - LAMDA_index), 8))
        finally_result.append(auto_value)
    finally_result = np.asarray(finally_result)
    return finally_result



def PseDNC(path):
    gene_type = 'DNA'
    fill_NA = '0'
    propertyname = r"physical_chemical_properties_DNA.txt"

    phisical_chemical_proporties = pd.read_csv(propertyname, header=None, index_col=None)
    m6a_sequence = open(path, 'r')
    DNC_key = phisical_chemical_proporties.values[:, 0]

    if fill_NA == "1":
        DNC_key[21] = 'NA'

    DNC_value = phisical_chemical_proporties.values[:, 1:]
    DNC_value = np.array(DNC_value).T
    DNC_value_scale = [[]] * len(DNC_value)
    # print (len(DNC_value))
    for i in range(len(DNC_value)):
        average_ = sum(DNC_value[i] * 1.0 / len(DNC_value[i]))
        std_ = np.std(DNC_value[i], ddof=1)
        DNC_value_scale[i] = [round((e - average_) / std_, 2) for e in DNC_value[i]]
        # print (DNC_value_scale)
    DNC_value_scale = list(zip(*DNC_value_scale))
    # print DNC_value_scale

    DNC_len = len(DNC_value_scale)
    # print (DNC_len)
    m6aseq = []
    for line in m6a_sequence:
        if line.startswith('>'):
            pass
        elif line == '\n':
            line = line.strip("\n")
        else:
            m6aseq.append(line.replace('\n', '').replace("\r", ''))
    w = 0.9
    Lamda = 2
    result_value = []
    m6a_len = len(m6aseq[0])
    # print m6a_len
    m6a_num = len(m6aseq)
    for m6a_line_index in range(m6a_num):
        frequency = [0] * len(DNC_key)
        # print len(frequency)
        m6a_DNC_value = [[]] * (m6a_len - 1)
        # print m6a_DNC_value
        for m6a_line_doublechar_index in range(m6a_len):
            for DNC_index in range(len(DNC_key)):
                if m6aseq[m6a_line_index][m6a_line_doublechar_index:m6a_line_doublechar_index + 2] == DNC_key[
                    DNC_index]:
                    # print m6aseq[2][0:2]
                    m6a_DNC_value[m6a_line_doublechar_index] = DNC_value_scale[DNC_index]
                    frequency[DNC_index] += 1
        # print m6a_DNC_value

        frequency = [e / float(sum(frequency)) for e in frequency]
        p = sum((frequency))
        # print p
        # frequency=np.array(frequency)/float(sum(frequency))#(m6a_len-1)
        one_line_value_with = 0.0
        sita = [0] * Lamda
        # print len(sita)
        for lambda_index in range(1, Lamda + 1):
            one_line_value_without_ = 0.0
            for m6a_sequence_value_index in range(1, m6a_len - lambda_index):
                temp = list(map(lambda x, y: round((x - y) ** 2, 8),
                                list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1])),
                                list(np.array(m6a_DNC_value[m6a_sequence_value_index - 1 + lambda_index]))))
                # map(lambda x, y: x + y, [1, 3, 5, 7, 9], [2, 4, 6, 8, 10])
                temp_value = round(sum(temp) * 1.0 / DNC_len, 8)
                one_line_value_without_ += temp_value
            one_line_value_without_ = round(one_line_value_without_ / (m6a_len - lambda_index - 1), 8)
            sita[lambda_index - 1] = one_line_value_without_
            one_line_value_with += one_line_value_without_
        dim = [0] * (len(DNC_key) + Lamda)
        # print len(dim)
        for index in range(1, len(DNC_key) + Lamda + 1):
            if index <= len(DNC_key):
                dim[index - 1] = frequency[index - 1] / (1.0 + w * one_line_value_with)
            else:
                dim[index - 1] = w * sita[index - len(DNC_key) - 1] / (1.0 + w * one_line_value_with)
            dim[index - 1] = round(dim[index - 1], 8)
        result_value.append(dim)
    # pd.DataFrame(result_value).to_csv(r'save file', header=None, index=None)
    finally_result = np.asarray(result_value)
    m6a_sequence.close()
    return finally_result



