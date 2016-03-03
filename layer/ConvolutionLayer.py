#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

import Layer
from activation import ActivatinFactory

class ConvolutionLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, kernel_num, kernel_width, kernel_height, bias, activation_type,
                 learning_rate):
        super(ConvolutionLayer, self).__init__(pre_data, post_data, learning_rate)
        self._kernel_num = kernel_num
        self._kernel_width = kernel_width
        self._kernel_height = kernel_height

        self._init_kernel()

        self._bias = bias
        self._activation = ActivatinFactory.get_activation(activation_type)


    def _init_kernel(self):
        low, high = -0.5, 0.5
        self._kernel = np.random.uniform(low, high, [self._kernel_num, self._pre_data.get_channel(),
                                                     self._kernel_height, self._kernel_width])

    def backward(self):
        pass

    def forward(self):
        '''
        convolve_and_set_post_data
        将data与kernel均展开成大矩阵做卷积
        所有通道的图片整合成一个大矩阵，每个kernel也整合成对应的矩阵， 乘self._kernel_size遍就可以了
        '''
        new_data, new_kernel = self._expand()
        result = np.dot(new_data, new_kernel)
        result += self._bias
        activation_func = np.vectorize(ConvolutionLayer._activate_elementwise_func)
        result = activation_func(result, self._activation)

        self._split_and_set_post_data(result)

    def _expand(self):
        '''
        将data整合成 ((pw-w+1)(ph-h+1)) * (hwc)的矩阵
        将kernel整合成 (hwc) * kernel_num 的矩阵
        :return:new_data, new_kernel
        '''
        pre_data_tensor = self._pre_data.get_data()
        pre_data_channel = self._pre_data.get_channel()
        pre_data_width, pre_data_height = self._pre_data.get_width_height()

        #For data_structure
        new_data = []
        for row in xrange(pre_data_height-self._kernel_height+1):
            for col in xrange(pre_data_width-self._kernel_width+1):
                data_row = []
                for channel in xrange(pre_data_channel):
                    for kernel_box_row in xrange(self._kernel_height):
                        for kernel_box_col in xrange(self._kernel_width):
                            data_row.append(pre_data_tensor[channel, row+kernel_box_row, col+kernel_box_col])
                new_data.append(np.array(data_row))
        new_data = np.array(new_data)

        #For kernel
        new_kernel = []
        for channel in xrange(pre_data_channel):
            for row in xrange(self._kernel_height):
                for col in xrange(self._kernel_width):
                    kernel_row = []
                    for kernel_ix in xrange(self._kernel_num):
                        kernel_row.append(self._kernel[kernel_ix, channel, row, col])
                    new_kernel.append(kernel_row)
        new_kernel = np.array(new_kernel)
        return new_data, new_kernel

    def _split_and_set_post_data(self, result):
        post_data_tensor = self._post_data.get_data()
        post_data_channel = self._post_data.get_channel()
        post_data_width, post_data_height = self._post_data.get_width_height()
        for kernel_ix in xrange(post_data_channel):
            ix = 0
            for row in xrange(post_data_height):
                for col in xrange(post_data_width):
                    post_data_tensor[kernel_ix, row, col] = result[ix, kernel_ix]
                    ix += 1

    @staticmethod
    def _activate_elementwise_func(element, func):
        return func(element)