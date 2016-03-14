#-*- coding: utf-8 -*-
__author__ = 'tao'

import Sigmod
import Linear

def get_activation(activation_type):
    result = None
    if not activation_type:
        raise ValueError("Pooling type should not be None")
    elif activation_type.lower() == 'logistic' or activation_type.lower() == 'sigmod':
        result = Sigmod.Sigmod()
    elif activation_type.lower() == 'linear':
        result = Linear.Linear()
    else:
        raise ValueError("Pooling type must be either average or max")

    return result