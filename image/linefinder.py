import cv2


def find_lines(img):
    edges = cv2.Canny(img, 100, 200)

    return edges
