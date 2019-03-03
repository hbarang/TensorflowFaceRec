
import cv2 as cv
import numpy as np
import imutils
from test_img_crop import  crop_from_video

def detect(cascade_loc='haarcascade_frontalface_alt_tree.xml'):
    face_casc = cv.CascadeClassifier(cascade_loc)
    videCapture = cv.VideoCapture("CanerVideo/WIN_20180727_13_43_17_Pro.mp4")

    if videCapture.isOpened() == False:
        print("error")
        return

    counter = 1

    while (videCapture.isOpened()):
        ret, frame = videCapture.read()
        faces = face_casc.detectMultiScale(frame, 1.3, 2)

        if ret == True:
            for (x, y, w, h) in faces:
                if counter % 4 == 0:
                    frame = frame[y:y+h, x:x+w]
                    r = 227.0 / frame.shape[1]
                    dim = (227, int(frame.shape[0] * r))
                    resized = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
                    cv.imwrite("CanerVideo/Caner" + str(int(counter/5)) + ".jpg", resized)
                    #crop_from_video(frame, (x, y, x+w, y+h), "Tests/Mehmet" + str(int(counter/5)))
                counter = counter + 1

        else:
            break
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    videCapture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    detect()