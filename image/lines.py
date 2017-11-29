import cv2
import numpy as np
import random

# CONSTANTS
EPSILON = 1


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
    _, contours, h = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
                # TODO: Closed shapes with knobs on them act strangely
                # TODO: Remove duplicated points from lines
                # TODO: Branch detection
                points = contour.squeeze()

                if not closed:
                    # Check for the two points where stuff starts looping back
                    # And grab the bits just before that
                    rolled = np.roll(points, 2, axis=0)
                    loop = rolled == points
                    indices = np.where(loop.any(axis=1))[0]
                    start = indices[0] - 1
                    end = indices[1]
                    points = points.take(range(start, end), axis=0, mode='wrap')

                points = cv2.approxPolyDP(points, EPSILON, closed).squeeze()

                yield {
                    'color': label,
                    'closed': closed,
                    'points': points.tolist(),
                }
