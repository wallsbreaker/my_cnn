#-*- coding: utf-8 -*-
__author__ = 'tao'

import Pooling

import numpy as np

class AveragePooling(Pooling.Pooling):
    def pooing(self, array):
        assert array.shape == (self._window_height, self._window_width)
        return np.mean(array)

    def get_unsample_matrix(self, width, height):
        size = width * height
        result = np.ones([height, width])
        result /= size
        return result