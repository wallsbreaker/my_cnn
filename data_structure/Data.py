#-*- coding: utf-8 -*-
__author__ = 'tao'

import numpy as np

class Data(object):
    def __init__(self,channel, width, height, is_input=False):
        self._channel = channel
        self._width = width
        self._height = height
        self._data = np.zeros([channel, height, width])
        self._sensitivity = np.zeros([channel, height, width])
        self._is_input = is_input

    def get_channel(self):
        return self._channel

    def get_width_height(self):
        return self._width, self._height

    def is_input(self):
        return self._is_input

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

    def set_sensitivity(self, sensitivity):
        assert sensitivity.shape == (self._channel, self._height, self._width)
        self._sensitivity = sensitivity

    def get_sensitivity(self):
        assert self._sensitivity != None
        result = self._sensitivity
        return result


    @staticmethod
    def output_matrix2vector(matrix):
        assert matrix.shape[1] == 1 and matrix.shape[2] == 1
        result = [x[0, 0] for x in matrix]
        return np.array(result)
