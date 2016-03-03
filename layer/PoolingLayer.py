#-*- coding: utf-8 -*-
__author__ = 'tao'

import Layer

from pooling import PoolingFactory

class PoolingLayer(Layer.Layer):
    def __init__(self, pre_data, post_data, window_width, window_height, type, learning_rate):
        super(PoolingLayer, self).__init__(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._window_width = window_width
        self._window_height = window_height
        self._pool = PoolingFactory.get_pool(window_width, window_height, type)

    def forward(self):
        pre_data_tensor = self._pre_data.get_data()
        post_data_tensor = self._post_data.get_data()
        post_data_channel = self._post_data.get_channel()
        post_data_width, post_data_height = self._post_data.get_width_height()
        for channel in xrange(post_data_channel):
            for row in xrange(post_data_height):
                for col in xrange(post_data_width):
                    post_data_tensor[post_data_channel, post_data_height, post_data_width] =\
                        self._pool.pooing(pre_data_tensor[channel, post_data_height*self._window_height:
                        (post_data_height+1)*self._window_height, post_data_width*self._window_width:
                        (post_data_width+1)*self._window_width])

    def backward(self):
        pass