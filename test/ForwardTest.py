#-*- coding: utf-8 -*-
__author__ = 'tao'

import CNN
from data_structure import DataImporter

if __name__ == '__main__':
    cnn = CNN.CNN(0.1, 100)
    cnn.add_input_layer(3, 48, 48)
    cnn.add_conv_layer(10, 7, 7, 1)
    cnn.add_pooling_layer(2, 2)
    cnn.add_conv_layer(15, 4, 4, 1)
    cnn.add_pooling_layer(2, 2)
    cnn.add_conv_layer(20, 4, 4, 1)
    cnn.add_pooling_layer(2, 2)
    cnn.add_conv_layer(40, 3, 3, 1)
    cnn.add_full_connected_layer(80, 1)
    cnn.add_full_connected_layer(9, 1)

    d = DataImporter.DataImporter()
    images, labels = d.loadData("../data/mastif_ts2010_train.pkl")
    cnn.train(images[:3], labels[:3])
