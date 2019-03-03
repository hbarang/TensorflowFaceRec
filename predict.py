import tflearn
import numpy as np
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

import tensorflow as tf
from PIL import Image


def resizeimg(arr):
    arr = np.reshape(arr, (-1, 128, 128, 3))
    return arr


def prediction(img):
    dataset = "rgb_faces.txt"

    images, labels = image_preloader(dataset, image_shape=(128, 128), mode='file', categorical_labels=True,
                                     normalize=True)

    preProc = tflearn.DataPreprocessing()
    preProc.add_custom_preprocessing(resizeimg)

    acc = tflearn.metrics.Accuracy()

    network = input_data(shape=[None, 128, 128, 3])
    '''data_preprocessing=preProc'''
    ''', data_preprocessing=img_prep,
                         data_augmentation=img_aug'''

    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool')

    network = conv_2d(network, 128, 3, activation='relu')
    network = conv_2d(network, 128, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool')

    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool')

    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = conv_2d(network, 512, 3, activation='relu')
    network = max_pool_2d(network, 2, strides=2, name='maxpool')

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)

    network = fully_connected(network, 5, activation='softmax')

    network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.load('./rgb_faces.tflearn')

    return model.predict(img)


if __name__ == '__main__':
    print(prediction("test644.jpg"))
