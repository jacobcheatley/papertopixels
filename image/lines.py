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
        # TODO: Possible problem - weird bits on the INSIDE of shapes
        for i, contour in enumerate(contours):
            if hierarchy[0][i][3] == -1:  # Ensure only "outer" contours - ones with no parents
                closed = np.asscalar(hierarchy[0][i][2] != -1)  # If this has a child, it must be a loop
                # TODO: Find points of discontinuity and reorder unique list
                # TODO: Closed shapes with knobs on them aren't identified correctly
                points = contour.squeeze().tolist()
                yield {
                    'color': label,
                    'len': contour.shape[0],
                    'uniq_len': np.unique(contour, axis=0).shape[0],
                    'closed': closed,
                    'points': points,
                }
