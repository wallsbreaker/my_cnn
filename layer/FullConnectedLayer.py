#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

from activation import ActivatinFactory
import Layer

class FullConnectedLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, post_dimen, bias, activation_type, learning_rate):
        super(FullConnectedLayer, self).__init__(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._post_dimen = post_dimen
        self._bias = bias
        self._activation = ActivatinFactory.get_activation(activation_type)
        self._init_w()

    def _init_w(self):
        low, high = -0.5, 0.5
        self._w = np.random.uniform(low, high, [self._pre_data.get_channel(), self._post_dimen])

    def forward(self):
        pre_data_array = self._pre_data.get_output()
        result = np.dot(pre_data_array, self._w)

        post_data_tensor = self._post_data.get_data()
        post_data_channel = self._post_data.get_channel()
        assert len(result) == post_data_channel
        for ix in xrange(post_data_channel):
            post_data_tensor[ix, 0, 0] = result[ix]

    def backward(self):
        pass
