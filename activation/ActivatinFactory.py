#-*- coding: utf-8 -*-
__author__ = 'tao'

import math

def logistic(x):
    return 1 / (1+math.exp(-x))

def get_activation(activation_type):
    result = None
    if not activation_type:
        raise ValueError("Pooling type should not be None")
    elif activation_type.lower() == 'logistic':
        result = logistic
    else:
        raise ValueError("Pooling type must be either average or max")

    return result