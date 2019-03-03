import tflearn
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from PIL import Image
import tensorflow as tf
import numpy as np

def ResizeImg(arr):
    arr = np.reshape(arr, (-1, 128, 128, 3))
    return arr


dataset = "flowerdata.txt"

images, labels = image_preloader(dataset, image_shape=(128, 128, 3), mode='file', categorical_labels=True, normalize=True)

'''
test_img = np.reshape(test_img, (-1, 128, 128, 1))
test_img_2 = np.reshape(test_img, (-1, 128, 128, 1))
test_img_3 = np.reshape(test_img, (-1, 128, 128, 1))
'''

preProc = tflearn.DataPreprocessing()
preProc.add_custom_preprocessing(ResizeImg)
#images = np.reshape(images, (-1, 128, 128, 1))
#labels = np.reshape(labels, (-1, 2))

'''
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation()'''

acc = tflearn.metrics.Accuracy()

network = input_data(shape=[None, 128, 128, 3], data_preprocessing=preProc)
''', data_preprocessing=img_prep,
                     data_augmentation=img_aug'''

network = conv_2d(network, 96, 11, strides=6, activation='relu',regularizer='L2',    weight_decay=0.0005,bias_init='uniform', trainable=True, restore=True)

network = max_pool_2d(network, 7, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu',regularizer='L2', weight_decay=0.0005,bias_init='uniform', trainable=True, restore=True)

network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='relu')
network = dropout(network, 0.7)
network = fully_connected(network, 3, activation='softmax')



network = regression(network, optimizer='momentum', loss='categorical_crossentropy')


model = tflearn.DNN(network, tensorboard_verbose=0)
model.load('./flower.tflearn')
#model.fit(images, labels, n_epoch=15, shuffle=True, show_metric=True, validation_set=0.1)
#model.save('flower.tflearn')

np.set_printoptions(3, suppress=True)


test_image = Image.open("ss-people-diversity-customers-1280x720.jpg")
test_image = np.reshape(test_image, (-1, 128, 128, 3))
print(model.predict(test_image))


