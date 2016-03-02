#-*- coding: utf-8 -*-
__author__ = 'tao'

import Layer

from pooling import PoolingFactory


class PoolingLayer(Layer):
    def __init__(self, pre_data, post_data, window_width, window_height, type, learning_rate):
        super(pre_data, post_data, learning_rate)
        self._pre_data = pre_data
        self._window_width = window_width
        self._window_height = window_height
        self._pool = PoolingFactory.get_pool(window_width, window_height, type)