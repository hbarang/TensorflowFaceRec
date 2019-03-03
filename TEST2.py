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
                test_img = crop_from_video_as_obj(frame, (x, y, x+w, y+h))
                test_img = np.reshape(test_img
                ,(-1, 128, 128, 3))
                label_prob = model.predict(test_img)
                print(label_prob)
                label = model.predict_label(test_img)[0][0]
                if float(np.max(label_prob)) < 0.75:
                    cv.putText(frame, "unknown", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
                elif label == 0:
                    cv.putText(frame, "Baran", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
                elif label == 1:
                    cv.putText(frame, "Caner", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
                elif label == 2:
                    cv.putText(frame, "Cigdem", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
                #elif label == 3:
                 #   cv.putText(frame, "Engin", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
                elif label == 3:
                    cv.putText(frame, "Mehmet", (x, y), cv.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0))
            cv.imshow('test', frame)
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow('test', frame)

    videCapture.release()
    cv.destroyAllWindows()




def ResizeImg(arr):
    arr = np.reshape(arr, (-1, 128, 128, 3))
    return arr


dataset = "rgb_faces.txt"

images, labels = image_preloader(dataset, image_shape=(128, 128, 3), mode='file', categorical_labels=True, normalize=True)

'''
test_img = np.reshape(test_img, (-1, 128, 128, 1))
test_img_2 = np.reshape(test_img, (-1, 128, 128, 1))
test_img_3 = np.reshape(test_img, (-1, 128, 128, 1))
'''

#preProc = tflearn.DataPreprocessing()
#preProc.add_custom_preprocessing(ResizeImg)
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

network = input_data(shape=[None, 128, 128, 3] )
'''data_preprocessing=preProc'''
''', data_preprocessing=img_prep,
                     data_augmentation=img_aug'''

network = conv_2d(network, 32, [4,4], activation='relu')
network = conv_2d(network, 48, [2, 2], activation='relu')
network = max_pool_2d(network, 5)
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='sgd', loss='categorical_crossentropy')

model = tflearn.DNN(network, tensorboard_verbose=0)

if(1 == 1):
    model.load('./TEST2.tflearn')
    model.fit(images, labels, n_epoch=20, shuffle=True, show_metric=True, validation_set=0.2)
    model.save('TEST2.tflearn')
else:
    model.load('./TEST2.tflearn')




np.set_printoptions(3, suppress=True)
detect("BaranVideo/WIN_20180727_11_27_40_Pro.mp4", model)


