import cv2


def pil_load_img(path):
    image = cv2.imread(path)
    return image