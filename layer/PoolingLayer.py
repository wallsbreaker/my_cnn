#-*- coding: utf-8 -*-
__author__ = 'tao'

from data_structure import Data
import Layer
from pooling import PoolingFactory
from activation import ActivationFactory

import numpy as np

class PoolingLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, window_width, window_height, pooling_type, activation_type, learning_rate):
        super(PoolingLayer, self).__init__(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._window_width = window_width
        self._window_height = window_height
        self._pool = PoolingFactory.get_pool(window_width, window_height, pooling_type)

        self._activation = ActivationFactory.get_activation(activation_type)

        self._init_coefficient()
        self._init_bias()

    def _init_coefficient(self):
        low, high = -0.5, 0.5
        self._coefficient = np.random.uniform(low, high, self._post_data.get_channel())

    def _init_bias(self):
        low, high = -0.5, 0.5
        self._bias = np.random.uniform(low, high, self._post_data.get_channel())

    def forward(self):
        '''
        初版：只是pooling加激活函数
        '''
        pre_data_tensor = self._pre_data.get_data()
        post_data_tensor = self._post_data.get_data()
        post_data_tensor.fill(0)
        post_data_channel = self._post_data.get_channel()
        post_data_width, post_data_height = self._post_data.get_width_height()
        for channel in xrange(post_data_channel):
            for row in xrange(post_data_height):
                for col in xrange(post_data_width):
                    begin_height = row*self._window_height
                    end_height = (row+1)*self._window_height
                    begin_width = col * self._window_width
                    end_width = (col+1) * self._window_width
                    pooing_result = self._pool.pooing(pre_data_tensor[channel,
                                                      begin_height:end_height, begin_width:end_width])

                    post_data_tensor[channel, row, col] = self._coefficient[channel] * pooing_result + self._bias[channel]
        post_data_tensor = self._activation.apply_activate_elementwise(post_data_tensor)
        self._post_data.set_data(post_data_tensor)

    def backward(self):
        #update sensitivity
        post_data = self._post_data.get_data()
        post_channel = self._post_data.get_channel()
        post_sensitivity = self._post_data.get_sensitivity()
        unsample_matrix = self._pool.get_unsample_matrix(self._window_width, self._window_height)
        upsample = np.kron(post_sensitivity, unsample_matrix)
        derivate_u = self._activation.apply_derivate_elementwise_from_output(post_data)
        pre_delta = np.multiply(upsample, derivate_u)
        pre_delta = np.array([self._coefficient[ix]*pre_delta for ix in xrange(post_channel)])
        self._pre_data.set_sensitivity(pre_delta)

        #update coefficient and b
        for channel in post_channel:
            self._bias[channel] -= self._learning_rate * np.sum(post_sensitivity[channel])
            self._coefficient -= self._learning_rate * np.sum(np.multiply(post_data, post_sensitivity))