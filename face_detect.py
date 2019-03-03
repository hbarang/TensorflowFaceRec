import cv2 as cv
import random
from test_img_crop import crop
import os

f = open('rgb_faces.txt', 'a+')

def detect(img_loc, name, cascade_loc='haarcascade_frontalface_alt_tree.xml'):
    face_casc = cv.CascadeClassifier(cascade_loc)

    img = cv.imread(img_loc)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_casc.detectMultiScale(gray, 1.1, 2)

    for(x, y, w, h) in faces:
        face_name = (name + str(random.randint(0, 10000)) + "_test.jpg")
        crop(img_loc, (x, y, x+w, y+h), name + "/" + face_name)
    cv.imshow('faces', img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def detect_test(img_loc):
    face_casc= cv.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

    img = cv.imread(img_loc)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    faces = face_casc.detectMultiScale(gray, 1.1, 2)
    for(x, y, w, h) in faces:
        crop_loc = "test" + str(random.randint(0, 1000)) + ".jpg"
        crop(img_loc, (x, y, x+w, y+h), crop_loc)
        return crop_loc

def detect_from_path(file_path, name, cl):
    face_casc = cv.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

    img = cv.imread(file_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_casc.detectMultiScale(gray, 1.1, 2)

    for(x, y, w, h) in faces:
        pic_id = random.randint(0, 1000)
        f.write(name + "/" + name + str(pic_id) + ".jpg" + "\t" + str(cl))
        crop(file_path, (x, y, x+w, y+h), name + "/" + name + str(pic_id) + ".jpg")


def write_as_rgb(file_path, name, cl, filename):

    img = cv.imread(file_path)
    pic_id = random.randint(0, 100000)
    f.write(name + "/" + filename + "\t" + str(cl) + "\n")


if __name__ == "__main__":
    for filename in os.listdir("MehmetVideo"):
        if filename.endswith(".jpg"):
            write_as_rgb(os.path.join("MehmetVideo", filename), "MehmetVideo", 2, filename)

  #  for i in range(1, 6):
   #     detect("Baran/baran" + str(i) + ".jpg", "Baran")
   # for i in range(1, 8):
    #    detect("Engin/engin" + str(i) + ".jpg", "Engin")
f.close()
