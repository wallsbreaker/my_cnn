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
        self._init_bias()

    def _init_w(self):
        low, high = -0.5, 0.5
        self._w = np.random.uniform(low, high, [self._post_dimen, self._pre_data.get_channel()])

    def _init_bias(self):
        low, high = -0.5, 0.5
        self._bias = np.random.uniform(low, high)

    def forward(self):
        pre_data_array = np.array([Data.Data.output_matrix2vector(self._pre_data.get_data())])
        result = np.dot(self._w, pre_data_array.T)
        result += self._bias
        result = self._activation.apply_activate_elementwise(result)

        post_data_tensor = self._post_data.get_data()
        post_data_tensor.fill(0)
        post_data_channel = self._post_data.get_channel()
        assert len(result) == post_data_channel
        for ix in xrange(post_data_channel):
            post_data_tensor[ix, 0, 0] = result[ix, 0]

    #全连接层：经典反向传播算法
    def backward(self):
        post_sensitivity = np.array([Data.Data.output_matrix2vector(self._post_data.get_sensitivity())])
        input_data = np.array([Data.Data.output_matrix2vector(self._pre_data.get_data())])
        output_data = np.array([Data.Data.output_matrix2vector(self._post_data.get_data())])
        #updatge sensitivities
        w_delta = np.dot(self._w.T, post_sensitivity.T)
        derivate_u = self._activation.apply_derivate_elementwise_from_output(output_data)
        pre_sensitivity = np.multiply(w_delta, derivate_u)
        self._pre_data.set_sensitivity(np.array([[x] for x in pre_sensitivity]))

        #更新_w
        delta_w = np.dot(post_sensitivity.T, input_data)
        self._w -= self._learning_rate * delta_w

        #更新 _bias
        self._bias -= self._learning_rate * pre_sensitivity