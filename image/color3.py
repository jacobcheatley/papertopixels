import numpy as np
import cv2
import image_config
from matplotlib import pyplot as plt

# CONSTANTS
EDGE = 20

edge_mask = np.zeros((image_config.RESOLUTION, image_config.RESOLUTION, 1), np.uint8)
cv2.rectangle(edge_mask, (EDGE-1, EDGE-1), (511-EDGE, 511-EDGE), 255, thickness=cv2.FILLED)

# colors = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)])  # b g r k
colors = np.array([(120, 255, 255), (60, 255, 255), (0, 255, 255), (0, 0, -128)])  # b g r k

WHITE = np.ones(3, np.uint8) * 255


def split_colors(img):
    def kmeans_pass(K):
        z = np.float32(img.reshape((-1, 3)))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        return cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    w, h, c = img.shape

    # First pass to determine background / foreground
    ret, label, center = kmeans_pass(2)
    bg_index = np.argmax(np.sum(center, axis=1))

    # Set all background and edge pixels to white
    img = img.reshape((w * h, c))
    img[label[:, 0] == bg_index, :] = WHITE
    img = img.reshape((w, h, c))
    img[edge_mask[..., 0] == 0] = WHITE

    # Second pass to get each separate channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hues = []
    sats = []
    vals = []
    for x in range(w):
        for y in range(h):
            if np.all(img[y, x] != WHITE):
                hues.append(img[y, x, 0])
                sats.append(img[y, x, 1])
                vals.append(img[y, x, 2])

    plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.title('Hue / Sat')
    plt.hist2d(hues, sats, bins=(180, 255))
    plt.subplot(132)
    plt.title('Hue / Val')
    plt.hist2d(hues, vals, bins=(180, 255))
    plt.subplot(133)
    plt.title('Sat / Val')
    plt.hist2d(sats, vals, bins=(255, 255))
    plt.show()

    # Need to bring "high red" back to low red
    img[np.logical_and(img[..., 0] > 150, img[..., 2] > 100), 0] = 15
    # Nothing works haHAA - red is a mess
    ret, label, center = kmeans_pass(5)
    res = cv2.cvtColor(np.uint8(center)[label.flatten()].reshape((w, h, c)), cv2.COLOR_HSV2BGR)

    cv2.imshow('Result', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    split_colors(cv2.imread('out/29/light.png'))
