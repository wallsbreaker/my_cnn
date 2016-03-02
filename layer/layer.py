#-*- coding: utf-8 -*-
__author__ = 'tao'

class Layer(object):
    def __init__(self, pre_data, post_data, learning_rate):
        self._pre_data = pre_data
        self._post_data = post_data
        self._learning_rate = learning_rate

    #to be overwrited
    def forward(self):
        pass

    #to be overwrited
    def forward(self):
        pass