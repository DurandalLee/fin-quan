# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# data set for network model
class NetModelData(Dataset):
    def __init__(self, train_set: np.array, test_set: np.array, unit_len):
        self.train_set = train_set
        self.test_set = test_set
        self.unit_len = unit_len

        self.train_unit_num = int(train_set.shape[0] - unit_len + 1)
        self.test_unit_num = int(test_set.shape[0] - unit_len + 1)

        data_dim = train_set.shape[-1]
        self.x_train = np.zeros((self.train_unit_num, unit_len - 1, data_dim))
        self.train_min_max = np.zeros((self.train_unit_num, data_dim, 2))
        self.y_train_un = np.zeros(self.train_unit_num)

    # 归一化
    @staticmethod
    def normalised(data: np.array):
        col_num = data.shape[1]
        # 保存数组中最小、最大值
        min_max = np.zeros((col_num, 2))
        norm_data = []

        # 每一列进行归一化处理
        for fun_i in range(col_num):
            tmp_col = data[:, fun_i]
            min_max[fun_i, 0] = min(tmp_col)
            tmp_col = tmp_col - min(tmp_col)

            min_max[fun_i, 1] = max(tmp_col)
            tmp_col = tmp_col / max(tmp_col)
            norm_data.append(tmp_col)
        norm_data = np.array(norm_data).T

        return norm_data, min_max

    # 反归一化
    @staticmethod
    def anti_normalised(nor_data: np.array, min_max):
        anti_data = []
        for i_data in nor_data:
            anti_data.append(i_data * min_max[:, 1] + min_max[:, 0])
        return np.array(anti_data)

    # 单元数据集归一化
    def normalised_dataset(self, source_data: np.array):
        # 生成数据集
        data_set = []
        for fun_i in range(source_data.shape[0] - self.unit_len + 1):
            data_set.append(source_data[fun_i:fun_i + self.unit_len])
        data_set = np.array(data_set).astype(float)
        y = data_set[:, -1, [0]]

        # 数据集元素个数、元素列数
        unit_num = data_set.shape[0]

        # 归一化数据
        normalised_set = []
        # 每个单元最大/小值
        min_max = np.zeros((unit_num, data_set.shape[2], 2))

        # 获取归一化数据
        for fun_i in range(unit_num):
            norm_tmp, min_max[fun_i] = self.normalised(data_set[fun_i])
            normalised_set.append(norm_tmp)
        normalised_set = np.array(normalised_set)

        # 返回值：数据集单元划分归一化、单元未归一化标签：价格以及成交量、单元集最大最小值
        return normalised_set, np.array(y), min_max

    # 窗口移动生成单个归一化数据组
    def produce_data_unit(self, item: int, dataset: np.array):
        unit_data = dataset[item: item + self.unit_len].astype(float)
        norm_data, min_max = self.normalised(unit_data)
        return unit_data[-1], norm_data, min_max

    # 训练数据集组数长度
    def __len__(self):
        return self.train_unit_num

    # 模型使用数据和标签
    def __getitem__(self, item):
        y_un, norm_data, min_max = self.produce_data_unit(item, self.train_set)
        self.train_min_max[item] = min_max
        self.y_train_un[item] = y_un[0]
        self.x_train[item] = norm_data[:-1]
        x = torch.from_numpy(norm_data[:-1])
        y = torch.from_numpy(norm_data[-1])

        return x, y

    # 模型生成特征值
    def feature_data(self, model, data_train: np.array):
        # 特征序列
        feature_seqs = []
        group_num = int(data_train.shape[0] / self.unit_len)

        for fun_i in range(group_num):
            tmp_group = data_train[fun_i * self.unit_len]
            tmp_group = torch.as_tensor(tmp_group, dtype=torch.float32)
            for tmp_i in range(self.unit_len):
                temp = model(tmp_group[np.newaxis, :, :])[0].detach().numpy()
                feature_seqs.append(temp)
                tmp_group = np.insert(tmp_group[1:], len(tmp_group) - 1, temp, axis=0)

        # [group_num*unit_num, data_dim]
        return np.array(feature_seqs)

    # 模型特征值反归一化
    def anti_normalization(self, nor_data: np.array, min_max):
        anti_data = []
        unit_len = self.unit_len

        group_num = int(nor_data.shape[0]/unit_len)
        for fun_i in range(group_num):
            nor_data_tmp = nor_data[fun_i * unit_len:(fun_i+1) * unit_len]
            anti_data.append(self.anti_normalised(nor_data_tmp, min_max[fun_i * unit_len]))
        anti_data = np.array(anti_data).reshape(-1, nor_data.shape[-1])

        return np.array(anti_data)

    # 通过模型生成训练、测试数据集特征值，并反归一化
    # 返回训练、测试数据集特征值、标签
    def feature_generate(self, model):
        x_test = []
        y_test_un = []
        test_min_max = []
        # 生成测试数据集
        for fun_i in range(self.test_unit_num):
            y_un, norm_data, min_max = self.produce_data_unit(fun_i, self.test_set)
            test_min_max.append(min_max)
            y_test_un.append(y_un[0])
            x_test.append(norm_data[:-1])

        # 生成训练、测试数据特征值
        feature_train = self.feature_data(model, self.x_train)
        feature_test = self.feature_data(model, np.array(x_test))

        # 训练、测试数据特征值反归一化
        feature_train_anti = self.anti_normalization(feature_train, self.train_min_max)
        feature_test_anti = self.anti_normalization(feature_test, test_min_max)

        return feature_train_anti, feature_test_anti, self.y_train_un, y_test_un


# processing data
class ModelData:
    # filename:data path
    # split: proportion of training data
    # unit_len: the number of days for a unit also for predict
    # col_list: names for data to be used
    def __init__(self, filename: str, split: float, unit_len: int, col_list: list):
        # 用于训练的股票数据（构造符合split数据长度）
        self.dataframe = pd.read_csv(filename)[col_list]
        d_value = int(self.dataframe.shape[0] * split % unit_len +
                      self.dataframe.shape[0] * (1 - split) % unit_len)
        self.dataframe = self.dataframe.loc[d_value:].reset_index(drop=True)
        # 每个单位数据包含的天数\预测天数
        self.unit_len = unit_len

        # 训练、测速数据划分
        self.i_split = int(self.dataframe.shape[0] * split)
        train_set = self.dataframe.get(col_list).values[:self.i_split]
        test_set = self.dataframe.get(col_list).values[self.i_split:]
        # 构建用于LSTM训练的数据集
        self.lstm_data = NetModelData(train_set, test_set, unit_len)

        # 训练、测试特征值反归一化
        self.feature_train = np.array([])
        self.feature_test = np.array([])
        # 训练、测试集对应标签
        self.ture_train = np.array([])
        self.ture_test = np.array([])

        # 预测数据隐状态分类
        self.hidden_train = np.array([])
        self.hidden_test = np.array([])

        # 线性回归训练数据和测试数据
        self.x_linear_train = pd.DataFrame()
        self.y_linear_train = pd.DataFrame()
        self.x_linear_test = pd.DataFrame()
        self.y_linear_test = pd.DataFrame()

    # 计算相关系数
    # unit_len 每个相关单元长度
    # cols 对应需要计算相关系数的列名字
    @staticmethod
    def calculate_correlation(unit_len: int, dataframe: pd.DataFrame, cols: list):
        corr = np.zeros(len(dataframe))
        # 完整周期长度数量
        unit_num = int(len(dataframe) / unit_len)
        # 残余未足够周期数据长度
        unit_remainder = len(dataframe) - unit_num * unit_len

        # 满堆情况
        for fun_i in range(unit_num):
            ind_stock = dataframe[cols[0]][fun_i * unit_len: fun_i * unit_len + unit_len].tolist()
            spy_stock = dataframe[cols[1]][fun_i * unit_len: fun_i * unit_len + unit_len].tolist()
            correlation = np.corrcoef(ind_stock, spy_stock)[0, 1]
            corr[fun_i * unit_len: fun_i * unit_len + unit_len] = correlation

        if unit_remainder != 0:
            # 不满堆情况
            ind_stock = dataframe[cols[0]][unit_num * unit_len: unit_num * unit_len + unit_remainder].tolist()
            spy_stock = dataframe[cols[1]][unit_num * unit_len: unit_num * unit_len + unit_remainder].tolist()
            correlation = np.corrcoef(ind_stock, spy_stock)[0, 1]
            corr[unit_num * unit_len: unit_num * unit_len + unit_remainder] = correlation

        return corr

    # create feature by model
    def feature_generate(self, model):
        self.feature_train, self.feature_test, self.ture_train, self.ture_test = \
            self.lstm_data.feature_generate(model)

    # 通过原数据产生HMM训练数据
    def hmm_data_origin(self, data_cols):
        # 训练、测速数据划分
        train_data = self.dataframe.get(data_cols).values[:self.i_split]
        test_data = self.dataframe.get(data_cols).values[self.i_split:]
        # 数据归一化
        train_data = train_data/np.max(train_data, axis=0)
        test_data = test_data/np.max(test_data, axis=0)

        return train_data, test_data

    # 原始数据隐状态隔周期对应（使用HMM模型）
    def hmm_state_origin(self, train_out: np.array, test_out: np.array):
        # 训练数据特征与实际数据长度差
        diff_len = train_out.shape[0] - self.feature_train.shape[0]
        train_out = train_out[: (train_out.shape[0] - diff_len)]
        self.hidden_train = np.squeeze(train_out)

        diff_len = test_out.shape[0] - self.feature_test.shape[0]
        test_out = test_out[: (test_out.shape[0] - diff_len)]
        self.hidden_test = np.squeeze(test_out)

    # 通过原数据产生HMM训练数据
    def hmm_data_feature(self, ):
        return self.feature_train[:, :2], self.feature_test[:, :2]

    # 原始数据隐状态隔周期对应（使用HMM模型）
    def hmm_state_feature(self, train_out: np.array, test_out: np.array):
        self.hidden_train = np.squeeze(train_out)
        self.hidden_test = np.squeeze(test_out)

    # 构建线性回归数据集
    def linear_regression_dataset(self, corr_cols):
        # 训练数据集构成
        # 数据部分
        linear_train = np.c_[self.feature_train, self.hidden_train]
        columns = ['price_fea', 'volume_fea', 'index_price_fea', 'hidden_class']
        linear_train = pd.DataFrame(linear_train, columns=columns)

        # 计算相关系数
        corr = self.calculate_correlation(self.unit_len, linear_train, corr_cols)
        linear_train['correlation'] = corr
        self.x_linear_train = linear_train

        # 标签部分
        self.y_linear_train = pd.DataFrame(self.ture_train[:linear_train.shape[0]])

        # 测试数据集构成
        # 数据部分
        linear_test = np.c_[self.feature_test, self.hidden_test]
        columns = ['price_fea', 'volume_fea', 'index_price_fea', 'hidden_class']
        linear_test = pd.DataFrame(linear_test, columns=columns)

        # 计算相关系数
        corr = self.calculate_correlation(self.unit_len, linear_test, corr_cols)
        linear_test['correlation'] = corr
        self.x_linear_test = linear_test

        # 标签部分
        self.y_linear_test = pd.DataFrame(self.ture_test)

        return self.x_linear_train, self.y_linear_train, self.x_linear_test, self.y_linear_test
