# DSNm6A
## Datasets
 - 包含五个物种的lncRNA和mRNA数据集
## Description
 - encode_method.py 特征编码方法
 - encode.py 生成序列特征编码
 - BERT.py BERT的网络模型
 - model_full.py CNN和Bi-LSTM的网络模型
 - model_pskp_jiu.py DSN的网络模型
 - function.py 损失函数
 - train_model_pskp_jiu.py 训练模型
 - test_pskp_jiu.py 测试模型性能
## Run Step
1. 运行encode.py 生成序列特征编码
2. 运行train_model_pskp_jiu.py 训练模型并测试模型性能
## Requirements
 - Python == 3.8
 - Numpy == 1.24.3
 - Pandas == 1.3.5
 - Pytorch == 1.12.0
 - Scikit-learn == 1.0.2
 - Torchmetrics == 0.10.1
