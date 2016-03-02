#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

import Layer
from activation import ActivatinFactory

class ConvolutionLayer(Layer):
    def __init__(self, pre_data, post_data, kernel_num, kernel_width, kernel_height, bias, activation_type, learning_rate):
        super(pre_data, post_data, learning_rate)
        self._kernel_num = kernel_num
        self._kernel_width = kernel_width
        self._kernel_height = kernel_height

        self._init_kernel()

        self._bias = bias
        self._activation = ActivatinFactory.get_activation(activation_type)


    def _init_kernel(self):
        low, high = -0.5, 0.5
        self._kernel = np.random.uniform(low, high, [self._kernel_num, self.pre_data.get_channel(),
                                                     self._kernel_height, self._kernel_width])

    def forward(self):
        data = self._pre_data.get_data()



    def forward(self):
        pass
