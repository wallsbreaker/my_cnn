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


    def set_data(self, data):
        assert (self._channel, self._height, self._width) == data.shape
        self._data = data

    def get_data(self):
        return self._data

    def set_error(self, error):
        assert error.shape == (self._channel, self._width, self._height)
        self._error = error

    def get_error(self):
        assert self._error
        result = self._error
        #拿到一次之后就置为None，防止多次梯度下降之间用浑
        self._error = None
        return result


    @staticmethod
    def output_matrix2vector(matrix):
        assert matrix.shape[1] == 1 and matrix.shape[2] == 2
        result = [x[0, 0] for x in matrix]
        return np.array(result)
