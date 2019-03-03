import tflearn
from tflearn.data_utils import image_preloader
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from PIL import Image
import tensorflow as tf
import numpy as np

from predict import prediction
from face_detect import detect_test
import cv2 as cv
import numpy as np
import imutils
from test_img_crop import  crop_from_video_as_obj

def detect(video_path, model, cascade_loc='haarcascade_frontalface_alt_tree.xml'):
    face_casc = cv.CascadeClassifier(cascade_loc)
    videCapture = cv.VideoCapture(video_path)

    if videCapture.isOpened() == False:
        print("error")
        return
    while (videCapture.isOpened()):
        ret, frame = videCapture.read()
        faces = face_casc.detectMultiScale(frame, 1.3, 2)
        if ret == True :
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                test_img = frame[y:y + h, x:x + w]
                r = 227.0 / test_img.shape[1]
                dim = (227, int(test_img.shape[0] * r))
                test_img = cv.resize(test_img, dim, interpolation=cv.INTER_AREA)
                if np.shape(test_img) != (227, 227, 3):
                    break
                test_img = np.reshape(test_img, (-1, 227, 227, 3))
                #test_img = np.reshape(test_img
                #,(-1, 227, 227, 3))
                label_prob = model.predict(test_img)
                print(label_prob)
                label = model.predict_label(test_img)[0][0]
                if float(np.max(label_prob)) < 0.60:
                    cv.putText(frame, "unknown", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
                elif label == 0:
                    cv.putText(frame, "Caner", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
                elif label == 1:
                    cv.putText(frame, "Engin", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
                elif label == 2:
                    cv.putText(frame, "Mehmet", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
                #elif label == 3:
                 #   cv.putText(frame, "Engin", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
                elif label == 5:
                    cv.putText(frame, "Mehmet", (x, y), cv.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0))
            cv.imshow('test', frame)
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow('test', frame)

    videCapture.release()
    cv.destroyAllWindows()




def ResizeImg(arr):
    arr = np.reshape(arr, (-1, 227, 227, 3))
    return arr


dataset = "rgb_faces.txt"

images, labels = image_preloader(dataset, image_shape=(227, 227, 227), mode='file', categorical_labels=True, normalize=True)

'''
test_img = np.reshape(test_img, (-1, 128, 128, 1))
test_img_2 = np.reshape(test_img, (-1, 128, 128, 1))
test_img_3 = np.reshape(test_img, (-1, 128, 128, 1))
'''

#preProc = tflearn.DataPreprocessing()
#preProc.add_custom_preprocessing(ResizeImg)
#images = np.reshape(images, (-1, 128, 128, 1))
#labels = np.reshape(labels, (-1, 2))


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation()
img_aug.add_random_blur()

acc = tflearn.metrics.Accuracy()

network = input_data(shape=[None, 227, 227, 3], data_augmentation=img_aug)


network = conv_2d(network, 96, 11, strides=4, activation='relu')

network = max_pool_2d(network, 3, strides=2)

network = tflearn.local_response_normalization(network)

network = conv_2d(network, 256, 5, activation='relu')

network = max_pool_2d(network, 3, strides=2)

network = tflearn.local_response_normalization(network)

network = conv_2d(network, 384, 3, activation='relu')

network = conv_2d(network, 384, 3, activation='relu')

network = conv_2d(network, 256, 3, activation='relu')

network = max_pool_2d(network, 3, strides=2)

network = tflearn.local_response_normalization(network)

network = fully_connected(network, 4096, activation='tanh')

network = dropout(network, 0.5)

network = fully_connected(network, 4096, activation='tanh')

network = dropout(network, 0.5)

network = fully_connected(network, 3, activation='softmax')

network = regression(network, optimizer='momentum',

                     loss='categorical_crossentropy',

                     learning_rate=0.001)
model = tflearn.DNN(network, tensorboard_verbose=0)


mode = None

if mode == 0:
    model.fit(images, labels, n_epoch=5, shuffle=True, show_metric=True, validation_set=0.1, batch_size=64)
    model.save('TEST_VG.tflearn')
elif mode == 1:
    model.load('./TEST_VG.tflearn')
    model.fit(images, labels, n_epoch=5, shuffle=True, show_metric=True, validation_set=0.1, batch_size=64)
    model.save('TEST_VG.tflearn')
else :
    model.load('./TEST_VG.tflearn')

#CanerVideo/WIN_20180727_13_43_17_Pro.mp4
#CigdemVideo/WIN_20180727_12_04_04_Pro.mp4
#EnginVideo/WIN_20180730_12_01_20_Pro.mp4
#


np.set_printoptions(3, suppress=True)
#detect("BaranVideo/WIN_20180727_11_27_40_Pro.mp4", model)
test = Image.open("MehmetVideo/Mehmet8.jpg")
test = test.resize((227, 227), Image.ANTIALIAS)
test = np.array(test)
test = np.reshape(test, (-1, 227, 227, 3))
print(model.predict(test))

