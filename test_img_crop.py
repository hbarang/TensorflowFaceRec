from PIL import Image, ImageOps
import numpy as np

def crop(img_path, cords, save_loc):
    image_obj = Image.open(img_path)
    cropped_image = image_obj.crop(cords)
    cropped_image = ImageOps.fit(cropped_image, (128, 128), Image.ANTIALIAS, 0, (0.5, 0.5))
  #  cropped_image.save(save_loc)

    cropped_image = cropped_image.convert('L')
    y = np.asarray(cropped_image.getdata(), dtype=np.float64).reshape((cropped_image.size[1], cropped_image.size[0]))
    y = np.asarray(y, dtype=np.uint8)
    w = Image.fromarray(y, mode='L')
    w.save(save_loc)


def crop_from_video(img, cords, save_loc):
    cropped_image = Image.fromarray(img)
    cropped_image = cropped_image.crop(cords)
    cropped_image = ImageOps.fit(cropped_image, (128, 128), Image.ANTIALIAS, 0, (0.5, 0.5))
    cropped_image.save(save_loc + ".jpg")
       
def crop_from_video_as_obj(img, cords):
    cropped_image = Image.fromarray(img)
    cropped_image = cropped_image.crop(cords)
    cropped_image = ImageOps.fit(cropped_image, (128, 128), Image.ANTIALIAS, 0, (0.5, 0.5))
    return cropped_image