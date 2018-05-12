import numpy as np
import cv2
import image_config

# CONSTANTS
EDGE = 10

edge_mask = np.zeros((image_config.RESOLUTION, image_config.RESOLUTION, 1), np.uint8)
cv2.rectangle(edge_mask, (EDGE-1, EDGE-1), (1023-EDGE, 1023-EDGE), 255, thickness=cv2.FILLED)

# colors = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)])  # b g r k
colors = np.array([(120, 255, 255), (60, 255, 255), (0, 255, 255), (0, 0, -128)])  # b g r k

WHITE = 255
BLACK = 0


def find_mask(img):
    w, h, c = img.shape

    # kmeans pass of k=2 to determine foreground
    z = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Foreground will be darker on average
    fg_index = np.argmin(np.sum(center, axis=1))

    # White out where foreground is
    res = np.zeros((w * h, 1), np.uint8)
    res[label == fg_index] = WHITE
    # Reblack it
    res = res.reshape((w, h, 1))
    res[edge_mask[..., 0] == 0] = BLACK

    # Do a morphological close to fix up rough patches
    res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    return res


if __name__ == '__main__':
    find_mask(cv2.imread('../misc/light.png'))
