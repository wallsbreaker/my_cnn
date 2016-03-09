#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

from layer import PoolingLayer
from layer import ConvolutionLayer
from layer import FullConnectedLayer

from data_structure import Data



class CNN(object):
    def __init__(self, learning_rate, max_iter):
        self._data = [] #每层的数据
        self._layers = [] #按顺序卷积层

        self._learning_rate = learning_rate
        self._max_iter = max_iter

    def train(self, data, label):
        assert self._layers
        assert data
        assert label
        assert len(data) == len(label)
        assert len(data[0]) == self._input_channel
        assert len(data[0][0]) == self._input_height
        assert len(data[0][0][0]) == self._input_width

        sample_size = len(data)

        if __debug__:
                predict_correct = 0.0
                pred_label = self.predict(data)
                for ix in xrange(sample_size):
                    if pred_label[ix] == label[ix]:
                        predict_correct += 1
                print "Accuracy before iteration: {0}".format(predict_correct/sample_size)

        seq = np.r_[0:sample_size]
        for iter in xrange(self._max_iter):
            np.random.shuffle(seq)

            for ix in xrange(sample_size):
                self._data[0].set_data(np.array(data[seq[ix]]))

                for layer in self._layers:
                    layer.forward()

                #TODO:最后输出层的误差在这里计算

                for layer in reversed(self._layers):
                    layer.backward()

            if __debug__:
                predict_correct = 0.0
                pred_label = self.predict(data)
                for ix in xrange(sample_size):
                    if pred_label[ix] == label[ix]:
                        predict_correct += 1
                print "Iteration {0} accuracy: {1}".format(iter, predict_correct/sample_size)

    def predict(self, data):
        assert self._layers
        assert data
        assert len(data[0]) == self._input_channel
        assert len(data[0][0]) == self._input_height
        assert len(data[0][0][0]) == self._input_width

        label = []
        sample_size = len(data)
        for ix in xrange(sample_size):
            self._data[0].set_data(np.array(data[ix]))
            for layer in self._layers:
                layer.forward()
            assert self._data[-1].is_output()
            y = Data.Data.output_matrix2vector(self._data[-1].get_data())
            label.append(np.argmax(y))
        return label

    #增加输入数据层
    def add_input_layer(self, channel, width, height):
        if self._data:
            raise ValueError("Sorry, input layer cannot be added again")
        input_layer = Data.Data(channel, width, height)
        self._data.append(input_layer)

        self._input_channel = input_layer.get_channel()
        self._input_width, self._input_height = input_layer.get_width_height()

    #增加卷积层
    def add_conv_layer(self, kernel_num, kernel_width, kernel_height, activation_type='logistic'):
        if not self._data:
            raise ValueError("Sorry, you must add input layer first")
        pre_width, pre_height = self._data[-1].get_width_height()
        if pre_width < kernel_width :
            raise ValueError("Kernel width is too large for pre layer")
        if pre_height < kernel_height :
            raise ValueError("Kernel height is too large for pre layer")

        output_width = pre_width - kernel_width + 1
        output_height = pre_height - kernel_height + 1
        output_data = Data.Data(kernel_num, output_width, output_height)
        self._data.append(output_data)

        conv_layer = ConvolutionLayer.ConvolutionLayer(self._data[-2], output_data, kernel_num, kernel_width, kernel_height,
                                      activation_type, self._learning_rate)
        self._layers.append(conv_layer)


    #增加全连接层
    def add_full_connected_layer(self, post_dimen, activation_type='logistic'):
        if not self._data:
            raise ValueError("Sorry, you must add input layer first")
        pre_data = self._data[-1]
        pre_width, pre_height = pre_data.get_width_height()
        if pre_width != 1 or pre_height != 1:
            raise ValueError("Pre layer must output 1*1 features before full connected layer")

        assert post_dimen != 0
        assert activation_type != None

        output_data = Data.Data(post_dimen, 1, 1)
        self._data.append(output_data)

        full_conn_layer = FullConnectedLayer.FullConnectedLayer(pre_data, output_data, post_dimen,
                                                                activation_type, self._learning_rate)
        self._layers.append(full_conn_layer)


    #增加pooling层
    def add_pooling_layer(self, window_width, window_height, type='average'):
        if not self._data:
            raise ValueError("Sorry, you must add input layer first")
        pre_width, pre_height = self._data[-1].get_width_height()
        if pre_width < window_width :
            raise ValueError("The pooling window's width is too large for pre layer")
        if pre_height < window_height :
            raise ValueError("The pooling window's height is too large for pre layer")
        if pre_width % window_width != 0:
            raise ValueError("The pooling window's width must be divided evenly by pre layer")
        if pre_height % window_height != 0:
            raise ValueError("The pooling window's height must be divided evenly by pre layer")

        output_width = pre_width / window_width
        output_height = pre_height / window_height
        output_data = Data.Data(self._data[-1].get_channel(), output_width, output_height)
        self._data.append(output_data)

        pooling_layer = PoolingLayer.PoolingLayer(self._data[-2], output_data, window_width, window_height, type, self._learning_rate)
        self._layers.append(pooling_layer)

