# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset


class TransformerData(Dataset):
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, predict_period: int):
        """
        :param data_set: data set for transformer
        :param unit_size: number of days for a unit
        """
        if predict_period % predict_size != 0:
            raise Exception("predict cycle must divide the predict size.")

        self.data_set = data_set
        self.unit_size = unit_size
        self.predict_size = predict_size
        self.predict_period = predict_period
        self.unit_number = int(data_set.shape[0] - unit_size + 1)
        self.data_columns = []

        data_size = data_set.shape[-1]
        self.max_set = np.zeros((self.unit_number, data_size))
        self.min_set = np.zeros((self.unit_number, data_size))

        self.feature = np.zeros(1)
        self.en_x = self.de_x = self.de_y = []
        self.anti_feature = pd.DataFrame()

    def create_dataset(self, column: list):
        """
        create dataset for encoder input(en_x),decoder input(de_x) and decoder output(y)
        :param column: specified data name
        """
        data_set = self.data_set.get(column).values
        self.data_columns = column

        # create unit data
        source_data = []
        for unit_i in range(self.unit_number):
            source_data.append(data_set[unit_i:unit_i + self.unit_size])
        source_data = np.array(source_data).astype(float)

        # normalized data set
        norm_data, self.max_set, self.min_set = max_min_normalised(source_data)
        # create encoder and decoder input data set
        self.en_x = norm_data[:, :-self.predict_size]
        self.de_x = self.de_y = norm_data[:, self.predict_size:]

    def __len__(self):
        return len(self.en_x)

    def __getitem__(self, item: int):
        return self.en_x[item], self.de_x[item], self.de_y[item]

    def generate_feature(self, model):
        """
        generate feature by model
        :param model: trained model
        """
        # feature sequence, all feature data
        feature_seq = []
        group_num = int(self.unit_number / self.predict_period)
        cycle_num = int(self.predict_period / self.predict_size)

        for unit_i in range(group_num):
            tmp_en = self.en_x[unit_i * self.predict_period]
            tmp_de = self.de_x[unit_i * self.predict_period]
            tmp_en = torch.as_tensor(tmp_en, dtype=torch.float32)
            tmp_de = torch.as_tensor(tmp_de, dtype=torch.float32)

            if torch.cuda.is_available():
                tmp_en, tmp_de = tmp_en.cuda(), tmp_de.cuda()

            for _ in range(cycle_num):
                tmp = model(tmp_en[np.newaxis, :, :], tmp_de[np.newaxis, :, :]
                            ).cpu().detach().numpy()[-self.predict_size:]
                feature_seq.append(tmp)

                tmp_en = np.insert(tmp_en.cpu()[self.predict_size:],
                                   tmp_en.shape[0] - self.predict_size, tmp, axis=0)
                tmp_de = np.insert(tmp_de.cpu()[self.predict_size:],
                                   tmp_de.shape[0] - self.predict_size, tmp, axis=0)

                if torch.cuda.is_available():
                    tmp_en = torch.as_tensor(tmp_en).cuda()
                    tmp_de = torch.as_tensor(tmp_de).cuda()

        self.feature = np.array(feature_seq).reshape(-1, len(self.data_columns))

    def generate_data(self):
        """
        according to feature,anti_normalized all data
        """
        generate_data = []
        group_num = int(self.unit_number / self.unit_size)

        for unit_i in range(group_num):
            feature_tmp = self.feature[unit_i * self.unit_size:(unit_i + 1) * self.unit_size]
            generate_data.append(anti_max_min_normalised(feature_tmp, self.max_set[unit_i], self.min_set[unit_i]))

        self.anti_feature = pd.DataFrame(
            np.array(generate_data).reshape(-1, self.feature.shape[-1]),
            columns=self.data_columns
        )


class ModelData:
    def __init__(self, data_set: pd.DataFrame, unit_size: int, predict_size: int, predict_period: int, real_label: str):
        """
        :param data_set: data set of all column
        :param unit_size: the number of days for a unit
        """
        self.data_set = data_set
        self.unit_size = unit_size

        # transformer data
        self.transformer_data = TransformerData(data_set, unit_size, predict_size, predict_period)

        # HMM state
        self.hidden_state = np.zeros(data_set[unit_size - predict_size:].shape[0])

        # correlation coefficient (between individual stock and market's price)
        self.correlation_values = np.zeros(data_set[unit_size - predict_size:].shape[0])

        # linear regression data
        self.linear_data = pd.DataFrame()

        # real data for this period
        self.real_data = np.array(data_set.get(real_label)[unit_size - predict_size:])

    def transformer_dataset(self, column: list):
        """
        :param column:specified data name
        """
        self.transformer_data.create_dataset(column)
        return self.transformer_data

    def hidden_state_model(self, model, column: list):
        """
        using model to generate hidden state between variable column
        :param model: model of hidden state
        :param column: specified data name
        """
        model_data = self.data_set.get(column).values
        self.hidden_state = model(model_data)

    def correlation(self, column: list):
        """
        calculate correlation between variable column
        :param column: specified data name
        """
        cor_data = self.transformer_data.anti_feature.get(column).values
        # complete cycle number
        unit_num = int(len(self.data_set) / self.unit_size)
        # insufficient cycle length
        remainder_num = len(self.data_set) - unit_num * self.unit_size
        unit_size = self.unit_size

        # complete situation
        for unit_i in range(unit_num):
            index = range(unit_i * unit_size, unit_i * unit_size + unit_size)
            self.correlation_values[index] = np.corrcoef(cor_data[0, index], cor_data[1, index])[0, 1]

        # insufficient situation
        if remainder_num != 0:
            index = range(unit_num * unit_size, unit_num * unit_size + remainder_num)
            self.correlation_values[index] = np.corrcoef(cor_data[0, index], cor_data[1, index])[0, 1]

    def linear_dataset(self):
        """
        create data for linear regression, include transformer feature data, hidden state data and correlation data
        """
        self.linear_data = pd.DataFrame(self.transformer_data.anti_feature)
        self.linear_data.insert(0, 'hidden state', self.hidden_state)
        self.linear_data.insert(0, 'correlation', self.correlation_values)

        return self.linear_data, self.real_data


class AllData:
    def __init__(self, filename: str, split: int, unit_size: int,
                 predict_size: int, predict_period: int, real_label: str):
        """
        :param filename: data part
        :param split: division ratio of train and test data
        :param unit_size: the number of days for a unit also for predict
        :param predict_period: predict days for whole model
        :param predict_size: predict days for each unit
        :param real_label: label for predict
        """
        # create the length of data can divisible by split
        self.dataframe = pd.read_csv(filename)
        d_len = self.dataframe.shape[0]
        v_len = math.ceil(d_len * split % unit_size + d_len * (1 - split) % unit_size)
        self.dataframe = self.dataframe.loc[v_len:].reset_index(drop=True)
        # length of train data
        split_index = int(self.dataframe.shape[0] * split)

        # model data for train and test
        self.train_data = ModelData(self.dataframe[:split_index], unit_size, predict_size, predict_period, real_label)
        self.test_data = ModelData(self.dataframe[split_index:], unit_size, predict_size, predict_period, real_label)


def max_min_normalised(data: np.array):
    """
    Maximum and minimum normalization
    :param data: data that needs to be normalized
    :return:
        max_set: maximum data in each unit
        min_set: maximum data in each unit
        normal_data: normalized data
    """
    unit_num = data.shape[0]

    max_set = []
    min_set = []
    normal_data = []

    for col_i in range(unit_num):
        data_i = data[col_i]

        min_set.append(np.min(data_i, axis=0))
        data_i = data_i - np.min(data_i, axis=0)
        max_set.append(np.max(data_i, axis=0))
        data_i = data_i / np.max(data_i, axis=0)

        normal_data.append(data_i)

    return np.array(normal_data), np.array(max_set), np.array(min_set)


def anti_max_min_normalised(norm_data: np.array, max_set: np.array, min_set: np.array):
    """
    Maximum and minimum anti_normalization
    :param norm_data: a unit data
    :param max_set: maximum data in norm_data
    :param min_set: minimum data in norm_data
    :return:
        anti_data: anti_normalised data
    """
    anti_data = []
    for col_i in range(norm_data.shape[0]):
        data_i = norm_data[col_i]
        data_i = data_i * max_set + min_set

        anti_data.append(data_i)
    return np.array(anti_data)
