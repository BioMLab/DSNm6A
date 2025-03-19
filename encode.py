import numpy as np
from encode_method import onehot, ncpf, pskp
import os

# # onehot编码
# filePath = 'dataset/DS_H/lncRNA/'
# files = os.listdir(filePath)
# for i in files:
#     print(i)
#     print(i[:-4])
#     encode = np.array(onehot(filePath + i))
#     np.save('encode/DS_H/lncRNA/onehot/' + i[:-4], encode)
#
# filePath = 'dataset/DS_H/mRNA/'
# files = os.listdir(filePath)
# for i in files:
#     print(i)
#     print(i[:-4])
#     encode = np.array(onehot(filePath + i))
#     np.save('encode/DS_H/mRNA/onehot/' + i[:-4], encode)
#
#
#
# # # ncpf编码
# filePath = 'dataset/DS_H/lncRNA/'
# files = os.listdir(filePath)
# for i in files:
#     print(i)
#     print(i[:-4])
#     encode = np.array(ncpf(filePath + i))
#     np.save('encode/DS_H/lncRNA/ncpf/' + i[:-4], encode)
#
# filePath = 'dataset/DS_H/mRNA/'
# files = os.listdir(filePath)
# for i in files:
#     print(i)
#     print(i[:-4])
#     encode = np.array(ncpf(filePath + i))
#     np.save('encode/DS_H/mRNA/ncpf/' + i[:-4], encode)
#
#
#
#
#
#
#
# # # pskp编码
# filePath = 'dataset/DS_H/lncRNA/'
# files = os.listdir(filePath)
# for i in files:
#     print(i)
#     print(i[:-4])
#     encode = np.array(pskp(filePath + i))
#     np.save('encode/DS_H/lncRNA/pskp/' + i[:-4], encode)

filePath = 'dataset/DS_H/mRNA/'
files = os.listdir(filePath)
for i in files:
    print(i)
    print(i[:-4])
    encode = np.array(pskp(filePath + i))
    np.save('encode/DS_H/mRNA/pskp/' + i[:-4], encode)



