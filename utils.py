# coding:utf-8
from __future__ import print_function

import torch.nn.init as init
import math
from torch.nn.utils import weight_norm

def adjust_learning_rate(optimizer, lr):
    """
    shrink learning rate for pytorch
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_weight_(weight):
    init.kaiming_uniform_(weight)


def init_embedding_(input_embedding):
    """
    Initialize embedding
    """
    # init_weight_(input_embedding.weight)
    init.normal_(input_embedding.weight, 0, 0.1)


def init_linear_(input_linear, in_features, dropout):
    """
    Initialize linear transformation
    """
    init.normal_(input_linear.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    # init_weight_(input_linear.weight)
    if input_linear.bias is not None:
        # input_linear.bias.dataset.zero_()
        init.constant_(input_linear.bias, 0)


def init_lstm_(input_lstm):
    """
    Initialize lstm
    """
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        init_weight_(weight)

        weight = eval('input_lstm.weight_hh_l' + str(ind))
        init_weight_(weight)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            init_weight_(weight)

            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            init_weight_(weight)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


def init_cnn_(input_cnn, kernel_size, in_channels, dropout):
    """
    Initialize cnn
    """
    std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * in_channels))
    init.normal_(input_cnn.weight, mean=0, std=std)
    init.constant_(input_cnn.bias, 0)


import torch


def log_sum_exp(x, dim=None, keepdim=False):
    """
    Calculate the log of the sum of the exponential of x, along dimension "dim"
    :param x: tensor
    :param dim: int, dimension index
    :param keepdim: bool, keep the size or not
    :return: log of the sum of the exponential of x, along dimension "dim"
    """
    if dim is None:
        x, dim = x.view(-1), 0
    xm, _ = torch.max(x, dim, keepdim=True)
    x = torch.where(
        (xm == float('inf')) | (xm == float('-inf')),
        xm,
        xm + torch.log(torch.sum(torch.exp(x - xm), dim, keepdim=True)))
    return x if keepdim else x.squeeze(dim)
