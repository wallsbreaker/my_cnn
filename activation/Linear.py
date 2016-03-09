__author__ = 'tao'

import Activation

class Linear(Activation.Activation):
    def activate(self, x):
        return x

    def derivative(self, x):
        return 1