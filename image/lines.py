import cv2
import numpy as np
import random


def random_color():
    return random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)


def show_contours(img, label='Contours'):
    _, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        cv2.drawContours(res, contours, i, random_color())
    cv2.imshow(label, res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_contours(img):
    _, contours, h = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contour_image(img, contours):
    res = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(len(contours)):
        cv2.drawContours(res, contours, i, random_color())
    return res


def get_all_lines(*img_and_labels):
    for img, label in img_and_labels:
        _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # TODO: A bunch more stuff with hierarchies
        # Possible problem - weird bits on the INSIDE of shapes
        closed = False
        for i, contour in enumerate(contours):
            # TODO: Loop detection
            if hierarchy[0][i][3] == -1:  # Ensure only "outer" contours
                yield {
                    'color': label,
                    'points': contour.squeeze().tolist(),
                    'closed': closed
                }
