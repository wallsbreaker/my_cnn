#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

from activation import ActivatinFactory
import Layer

class FullConnectedLayer(Layer):
    def __init__(self, pre_data, post_data, post_dimen, bias, activation_type, learning_rate):
        super(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._post_dimen = post_dimen
        self._bias = bias
        self._activation = ActivatinFactory.get_activation(activation_type)
        self._w = self._init_w()

    def _init_w(self):
        low, high = -0.5, 0.5
        self._kernel = np.random.uniform(low, high, [self._pre_data.get_channel(), self._pre_dimen])

