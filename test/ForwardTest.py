#-*- coding: utf-8 -*-
__author__ = 'tao'

#import CNN

class Car(object):
    def __init__(self, model, color, mpg):
        self.model = model
        self.color = color
        self.mpg   = mpg

class ElectricCar(Car):
    def xx(self):
        pass
if __name__ == '__main__':
    #cnn = CNN(0.1, 100)
    #cnn.add_input_layer(3, 48, 48)
    a = ElectricCar(0, 0, 0)