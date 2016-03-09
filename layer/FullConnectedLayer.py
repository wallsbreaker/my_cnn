#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

from activation import ActivationFactory
import Layer
from data_structure import Data

class FullConnectedLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, post_dimen, activation_type, learning_rate):
        super(FullConnectedLayer, self).__init__(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._post_dimen = post_dimen
        self._activation = ActivationFactory.get_activation(activation_type)
        self._init_w()

        #

    def _init_w(self):
        low, high = -0.5, 0.5
        self._w = np.random.uniform(low, high, [self._post_dimen, self._pre_data.get_channel()])


    def forward(self):
        pre_data_array = self._pre_data.get_output()
        result = np.dot(self._w, pre_data_array)
        result = self._activation.apply_activate_elementwise(result)

        post_data_tensor = self._post_data.get_data()
        post_data_channel = self._post_data.get_channel()
        assert len(result) == post_data_channel
        for ix in xrange(post_data_channel):
            post_data_tensor[ix, 0, 0] = result[ix]

    #全连接层：经典反向传播算法
    def backward(self):
        #更新w
        output_error = np.array([Data.Data.output_matrix2vector(self._post_data.get_error())])
        input_data = np.array([Data.Data.output_matrix2vector(self._pre_data.get_data())])
        delta_w = np.dot(output_error.T, input_data)
        self._w -= delta_w

        #根据上一层的error更新error
        delta_pre_error = np.multiply(np.dot(self._w.T, output_error.T),
                                      self._activation.apply_derivate_eleementwise(input_data.T))
        self._pre_data.set_error(np.array([[x] for x in delta_pre_error]))