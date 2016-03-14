__author__ = 'tao'

import Activation

import math

class Sigmod(Activation.Activation):
    def activate(self, x):
        #TODO
        if x < -100:
            x = -100
        return 1.0/(1+math.exp(-x))

    def derivative(self, x):
        result = self.activate(x)
        return result*(1-result)