# -*- coding: utf-8 -*-
from abc import ABC
import math
import numpy as np
import torch
import torch.nn as nn


def subsequence_mask(seq):
    attn_shape = [seq.shape[0], seq.shape[1], seq.shape[1]]
    mask = np.triu(np.ones(attn_shape), k=1)
    mask = torch.from_numpy(mask).bool()
    return mask


class PositionalEncoding(nn.Module, ABC):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].size()[1]]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[1]]
        return self.dropout(x)


class MultiHeadAttention(nn.Module, ABC):
    def __init__(self, data_dim, kq_dim, v_dim, heads_num):
        super(MultiHeadAttention, self).__init__()
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.heads_num = heads_num
        self.matrix_Q = nn.Linear(data_dim, kq_dim * heads_num, bias=False)
        self.matrix_K = nn.Linear(data_dim, kq_dim * heads_num, bias=False)
        self.matrix_V = nn.Linear(data_dim, v_dim * heads_num, bias=False)
        self.fully_con = nn.Linear(v_dim * heads_num, data_dim, bias=False)

    def forward(self, mask, *input_data):
        if len(input_data) == 1:
            input_q = input_k = input_v = input_data[0]
        else:
            input_q = input_k = input_data[0]
            input_v = input_data[1]

        batch_size = input_q.shape[0]

        q_vector = self.matrix_Q(input_q).reshape(batch_size, -1, self.heads_num, self.kq_dim).transpose(1, 2)
        k_vector = self.matrix_K(input_k).reshape(batch_size, -1, self.heads_num, self.kq_dim).transpose(1, 2)
        v_vector = self.matrix_V(input_v).reshape(batch_size, -1, self.heads_num, self.v_dim).transpose(1, 2)

        scores = torch.matmul(q_vector, k_vector.transpose(-1, -2)) / np.sqrt(self.kq_dim)

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.heads_num, 1, 1)
            scores.masked_fill_(mask, -1e9)

        scores = nn.Softmax(dim=-1)(scores)
        z_vector = torch.matmul(scores, v_vector)
        z_vector = z_vector.transpose(1, 2).reshape(batch_size, -1, self.heads_num * self.v_dim)
        output_data = self.fully_con(z_vector)

        return output_data


class EncoderLayer(nn.Module, ABC):
    def __init__(self, data_dim, kq_dim, v_dim, heads_num, forward_dim):
        super(EncoderLayer, self).__init__()
        self.encoder_multi_attention = MultiHeadAttention(data_dim, kq_dim, v_dim, heads_num)
        self.feedforward = nn.Sequential(
            nn.Linear(data_dim, forward_dim, bias=False),
            nn.ReLU(),
            nn.Linear(forward_dim, data_dim, bias=False),
        )
        self.layer_norm = nn.LayerNorm(data_dim, elementwise_affine=False)

    def forward(self, encoder_input):
        attn_output = self.encoder_multi_attention(None, encoder_input)
        layer_output = self.layer_norm(encoder_input + attn_output)
        feed_output = self.feedforward(layer_output)
        encoder_output = self.layer_norm(layer_output + feed_output)

        return encoder_output


class DecoderLayer(nn.Module, ABC):
    def __init__(self, data_dim, kq_dim, v_dim, heads_num, forward_dim):
        super(DecoderLayer, self).__init__()
        self.decoder_multi_attention = MultiHeadAttention(data_dim, kq_dim, v_dim, heads_num)
        self.en_de_multi_attention = MultiHeadAttention(data_dim, kq_dim, v_dim, heads_num)
        self.feedforward = nn.Sequential(
            nn.Linear(data_dim, forward_dim, bias=False),
            nn.ReLU(),
            nn.Linear(forward_dim, data_dim, bias=False),
        )
        self.layer_norm = nn.LayerNorm(data_dim, elementwise_affine=False)

    def forward(self, encoder_output, decoder_input, mask):
        self_attn_output = self.decoder_multi_attention(mask, decoder_input)
        self_layer_output = self.layer_norm(self_attn_output + decoder_input)
        attn_output = self.en_de_multi_attention(None, encoder_output, self_layer_output)
        layer_output = self.layer_norm(attn_output + self_layer_output)
        feed_output = self.feedforward(layer_output)
        decoder_output = self.layer_norm(layer_output + feed_output)

        return decoder_output


class Transformer(nn.Module, ABC):
    def __init__(self, dropout, data_size, kq_dim, v_dim, forward_dim,
                 en_layer, de_layer, en_head, de_head):
        super(Transformer, self).__init__()

        self.pos_encoding = PositionalEncoding(data_size, dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(data_size, kq_dim, v_dim, en_head, forward_dim) for _ in range(en_layer)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(data_size, kq_dim, v_dim, de_head, forward_dim) for _ in range(de_layer)
        ])
        self.fully_con = nn.Linear(data_size, data_size, bias=False)

    def forward(self, en_input, de_input):
        en_input = self.pos_encoding(en_input)
        de_input = self.pos_encoding(de_input)

        if torch.cuda.is_available():
            decoder_mask = subsequence_mask(de_input).cuda()
        else:
            decoder_mask = subsequence_mask(de_input)

        for layer in self.encoder_layers:
            en_input = layer(en_input)
        en_output = en_input

        for layer in self.decoder_layers:
            de_input = layer(en_output, de_input, decoder_mask)

        de_output = self.fully_con(de_input).reshape(-1, de_input.size(-1))
        de_output = nn.Softmax(dim=-1)(de_output)

        return de_output
