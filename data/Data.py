#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

class Data(object):
    def __init__(self,channel, width, height):
        self._channel = channel
        self._width = width
        self._height = height
        self._data = np.zeros([channel, width, height])

    def get_channel(self):
        return self._channel

    def get_width_height(self):
        return self._width, self._height

    def is_output(self):
        if self._width == 1 or self._height == 1:
            return True
        else:
            return False

    def get_output(self):
        result = [x[0, 0] for x in self._data]
        return np.array(result)

    def set_input(self, data):
        assert (self._channel, self._height, self._width) == data.shape
        self._data = data

    def get_data(self):
        return self._data