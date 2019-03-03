from predict import prediction
from face_detect import detect_test
import cv2 as cv
import numpy as np
import imutils
from test_img_crop import  crop_from_video_as_obj

def detect(video_path, cascade_loc='haarcascade_frontalface_alt_tree.xml'):
    face_casc = cv.CascadeClassifier(cascade_loc)
    videCapture = cv.VideoCapture(video_path)

    if videCapture.isOpened() == False:
        print("error")
        return
    counter = 1
    while (videCapture.isOpened()):
        ret, frame = videCapture.read()
        faces = face_casc.detectMultiScale(frame, 1.3, 2)
        if ret == True :
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                test_img = crop_from_video_as_obj(frame, (x, y, x+w, y+h))
                prediction(test_img)
            cv.imshow('test', frame)
        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        cv.imshow('test', frame)

    videCapture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect("BaranVideo/WIN_20180727_11_27_40_Pro.mp4")