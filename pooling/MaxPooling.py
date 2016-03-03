#-*- coding: utf-8 -*-
__author__ = 'tao'

import Pooling

import numpy as np

class MaxPooling(Pooling.Pooling):
    def pooling(self, array):
        assert array.shape == (self._window_height, self._window_width)
        return np.max(array)