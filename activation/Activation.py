__author__ = 'tao'

import numpy as np

class Activation(object):
    def activate(self, x):
        pass

    def derivative(self, result):
        pass

    def apply_activate_elementwise(self, data):
        func = np.vectorize(self.activate)
        result = func(data)
        return result

    def apply_derivate_elementwise(self, data):
        func = np.vectorize(self.derivative)
        result = func(data)
        return result