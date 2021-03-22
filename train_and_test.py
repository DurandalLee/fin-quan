# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import os
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression

import ModelSet
import DataClass
# 进程
# from multiprocessing import Pool


# train mid_lstm model
def train_lstm_model(lstm_model, train_batches, epochs, learning):
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning)
    loss_func = torch.nn.MSELoss()

    for epoch in range(epochs):
        for (x, y) in train_batches:
            x = torch.as_tensor(x, dtype=torch.float32)
            y = torch.as_tensor(y, dtype=torch.float32)
            y_predict = lstm_model(x)
            loss = loss_func(y_predict, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return lstm_model


# create hmm model and decode state
def hmm_state(para, data_set):
    # HMM train
    model_hmm = ModelSet.GaussianHMM(para['n_state'], para['x_size'], para['iter_time'])
    model_hmm.train(data_set)

    # decode hidden state
    hidden_state = np.array(model_hmm.decode(data_set))
    return hidden_state


# product train data for linear regression
def product_linear_data(data: pd.DataFrame):
    train_data = pd.DataFrame()
    train_data['correlation'] = data['correlation']
    train_data['price_fea'] = data['price_fea']
    train_data['hidden_class'] = data['hidden_class']
    train_data['corr_mul_index'] = data['correlation'] * data['index_price_fea']
    return train_data


# create linear regression model and predict values
def linear_regression(x_train, y_train, x_test):
    model_linear = LinearRegression().fit(x_train, y_train)
    predict = model_linear.predict(x_test)
    return predict


# point multiplication of sequence
def dot_product(vector1, vector2):
    return sum(np.array(vector1)*np.array(vector2))


# cosine similarity
def cos_sim(pre_data, true_data):
    return dot_product(pre_data, true_data)/math.sqrt(
        dot_product(pre_data, pre_data) * dot_product(true_data, true_data))


# percentage of error and accuracy
def percentage(pre_data, true_data):
    pre_accuracy = []
    for fun_i in range(len(pre_data)):
        error = abs((true_data[fun_i] - pre_data[fun_i]) / true_data[fun_i])
        pre_accuracy.append(1 - error)

    return pre_accuracy


# evaluate result
def evaluate(predict, true):
    cosine_sim = cos_sim(predict, true)
    pearson_cor = np.corrcoef(predict.flatten(), true.flatten())[0, 1]
    pre_accuracy = percentage(predict, true)
    return cosine_sim, pearson_cor, pre_accuracy


# create and save single stock model data
def single_stock_data(path, split, batch, unit_len, lstm_para, hmm_para):
    stock_name = path.split('/')[2].split('.')[0]
    print("Start mission " + stock_name)

    # 用于模型列名
    data_col_list = ['Close_y', 'Volume_y', 'Close_x']
    stock_data = DataClass.ModelData(path, split, unit_len, data_col_list)

    # 训练数据成批封装成Tensor类型
    train_batches = DataLoader(
        dataset=stock_data.lstm_data,
        batch_size=batch,
        pin_memory=True
    )
    # 创建lstm模型
    lstm_model = ModelSet.MidLSTM(
        input_size=lstm_para['input_size'],
        hidden_size=lstm_para['hidden_size'],
        batch_first=lstm_para['batch_first'],
        drop=lstm_para['drop']
    )
    # 调至训练模式
    lstm_model.train()
    # 训练LSTM模型
    lstm_model = train_lstm_model(
        lstm_model=lstm_model,
        train_batches=train_batches,
        epochs=lstm_para['epochs'],
        learning=lstm_para['learning_rate']
    )
    # 调至评估模式
    lstm_model.eval()
    # 生成特征值
    stock_data.feature_generate(lstm_model)

    # 使用原始数据训练HMM模型
    hmm_train, hmm_test = stock_data.hmm_data_origin(['Close_y', 'Volume_y'])
    # 保存HMM模型输出特征值
    hmm_train_out = hmm_state(hmm_para, hmm_train)
    hmm_test_out = hmm_state(hmm_para, hmm_test)
    stock_data.hmm_state_origin(hmm_train_out, hmm_test_out)

    # 生成线性回归数据
    x_train_ori, y_train, x_test_ori, y_test = \
        stock_data.linear_regression_dataset(['price_fea', 'index_price_fea'])
    x_train = product_linear_data(x_train_ori)
    x_test = product_linear_data(x_test_ori)
    y_test_pre = linear_regression(x_train, y_train, x_test)

    # 评价模型好坏程度
    stock_evaluate = evaluate(y_test, y_test_pre)
    print(stock_evaluate)

    mid_model_result = pd.DataFrame({'stock_name': [stock_name],
                                     'x_train': x_train_ori,
                                     'y_train': y_train,
                                     'x_test': x_test_ori,
                                     'y_test': y_test,
                                     'y_test_predict': np.arrau(y_test_pre)})

    save_path = './'+data_path.split('/')[1]+'_Result/' + stock_name + '.npy'
    np.save(save_path, mid_model_result)
    print("Finish mission " + stock_name)


# combine all stock data
def combine_all(sava_path, file_set):
    # 合并所有股票结果
    data_df = pd.DataFrame()
    for name in file_set:
        data_df = data_df.append(pd.DataFrame(np.load(name, allow_pickle=True)))
    np.save(sava_path, data_df)
    return data_df


# 数据文件参数
data_path = "./ShangHai/"

# 模型参数
split_percent = 0.85
batch_len = 60
unit_length = 60
lstm_parameter = {'input_size': 3, 'hidden_size': 60, 'batch_first': True,
                  'drop': 0.2, 'epochs': 2, 'learning_rate': 0.02}
hmm_parameter = {'x_size': 2, 'n_state': 4, 'iter_time': 10}

if __name__ == '__main__':
    # # 具有允许以几种不同方式将任务分配到工作进程的方法。
    # p = Pool(pool_num)
    # # 传入函数和函数使用参数
    # p.apply_async(function,args=())
    # p.close()  # 不再添加进程
    # p.join()  # 等待所有进程结束

    data_list = os.listdir(data_path)
    # 启动
    for path_i in data_list:
        single_stock_data(
            path=data_path + path_i,
            split=split_percent,
            batch=batch_len,
            unit_len=unit_length,
            lstm_para=lstm_parameter,
            hmm_para=hmm_parameter
        )
