#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np
from scipy import signal as sg

import Layer
from activation import ActivationFactory

class ConvolutionLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, kernel_num, kernel_width, kernel_height, activation_type,
                 learning_rate):
        super(ConvolutionLayer, self).__init__(pre_data, post_data, learning_rate)
        self._kernel_num = kernel_num
        self._kernel_width = kernel_width
        self._kernel_height = kernel_height

        self._init_kernel()
        self._init_bias()
        self._connected_table = np.ones([kernel_num, self._pre_data.get_channel()])

        self._activation = ActivationFactory.get_activation(activation_type)


    def _init_kernel(self):
        low, high = -0.5, 0.5
        self._kernel = np.random.uniform(low, high, [self._kernel_num, self._pre_data.get_channel(),
                                                     self._kernel_height, self._kernel_width])

    def _init_bias(self):
        low, high = -0.5, 0.5
        self._bias = np.random.uniform(low, high, self._kernel_num)

    def forward(self):
        '''
        convolve_and_set_post_data
        将data与kernel均展开成大矩阵做卷积
        所有通道的图片整合成一个大矩阵，每个kernel也整合成对应的矩阵， 乘self._kernel_size遍就可以了

        new_data, new_kernel = self._expand()
        result = np.dot(new_data, new_kernel)
        result += self._bias
        result = self._activation.apply_activate_elementwise(result)

        self._split_and_set_post_data(result)
        '''
        """
        按部就班
        """
        pre_data_tensor = self._pre_data.get_data()
        post_data_tensor = self._post_data.get_data()
        post_data_tensor.fill(0)
        for kernel in xrange(self._kernel_num):
            for channel in xrange(self._pre_data.get_channel()):
                if self._connected_table[kernel, channel]:
                    post_data_tensor[kernel] += sg.convolve2d(pre_data_tensor[channel], self._kernel[kernel, channel].T, 'valid')
            post_data_tensor[kernel] += self._bias[kernel]
            post_data_tensor[kernel] = self._activation.apply_activate_elementwise(post_data_tensor[kernel])
        self._post_data.set_data(post_data_tensor)

    def backward(self):
        #update kernel and bias
        pre_data_tensor = self._pre_data.get_data()
        pre_data_chanel = pre_data_tensor.shape[0]
        post_sensitivity = self._post_data.get_sensitivity()
        for kernel in xrange(self._kernel_num):
            for channel in xrange(pre_data_chanel):
                if self._connected_table[kernel, channel]:
                    delta_kernel = sg.convolve2d(pre_data_tensor[channel], post_sensitivity[kernel], 'valid')
                    self._kernel[kernel][channel] -= self._learning_rate * delta_kernel
            self._bias[kernel] -= self._learning_rate * np.sum(post_sensitivity[kernel])

        #update pre sensitivity
        pre_sensitivity = self._pre_data.get_sensitivity()
        pre_sensitivity.fill(0)
        for channel in xrange(pre_data_chanel):
            for kernel in xrange(self._kernel_num):
                if self._connected_table[kernel, channel]:
                    pre_sensitivity[channel] += sg.convolve2d(post_sensitivity[kernel], self._kernel[kernel][channel],
                            'full')
            derivate_u = self._activation.apply_derivate_elementwise_from_output(pre_data_tensor[channel])
            pre_sensitivity[channel] = np.multiply(pre_sensitivity[channel], derivate_u)
        self._pre_data.set_sensitivity(pre_sensitivity)


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
        post_data_channel, post_data_width, post_data_height = post_data_tensor.shape
        for kernel_ix in xrange(post_data_channel):
            ix = 0
            for row in xrange(post_data_height):
                for col in xrange(post_data_width):
                    post_data_tensor[kernel_ix, row, col] = result[ix, kernel_ix]
                    ix += 1
