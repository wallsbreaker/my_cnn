#-*- coding: utf-8 -*-
__author__ = 'tao'

import AveragePooling
import MaxPooling

def get_pool(window_width, window_height, type):
    result = None
    if not type:
        raise ValueError("Pooling type should not be None")
    elif type.lower() == 'average' or type.lower() == 'avg':
        result = AveragePooling.AveragePooling(window_width, window_height)
    elif type.low == 'max':
        result = MaxPooling.MaxPooling(window_width, window_height)
    else:
        raise ValueError("Pooling type must be either average or max")

    return result
