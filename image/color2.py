import numpy as np
import cv2

# CONSTANTS
EDGE = 6

edge_mask = np.zeros((512, 512, 1), np.uint8)
cv2.rectangle(edge_mask, (EDGE-1, EDGE-1), (511-EDGE, 511-EDGE), 255, thickness=cv2.FILLED)

# colors = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)])  # b g r k
colors = np.array([(120, 255, 255), (60, 255, 255), (0, 255, 255), (0, 0, -128)])  # b g r k

DEG_CONV = 2 * np.pi / 180


def lowest_distance(pixel):
    def dist(color):
        return (color[2] - pixel[2]) ** 2 + \
               (color[1] * np.cos(color[0] * DEG_CONV) - pixel[1] * np.cos(pixel[0] * DEG_CONV)) ** 2 + \
               (color[1] * np.sin(color[0] * DEG_CONV) - pixel[1] * np.sin(pixel[0] * DEG_CONV)) ** 2

    return np.argmin([dist(color) for color in colors])


def split_colors(img):
    z = np.float32(img.reshape((-1, 3)))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2

    ret, label, center = cv2.kmeans(z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    bg_index = np.argmax(np.sum(center, axis=1))
    thres = np.array([255, 0]) if bg_index == 1 else np.array([0, 255])
    res = np.uint8(thres[label.flatten()].reshape(img.shape[0], img.shape[1]))
    res = cv2.bitwise_and(res, edge_mask)

    cv2.imshow('result', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    b = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    g = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    r = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    k = np.zeros((img.shape[0], img.shape[0]), dtype=np.uint8)
    result = [b, g, r, k]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    for y in range(img.shape[0]):
        for x in range(img.shape[0]):
            if res[y, x]:
                result[lowest_distance(img[y, x])][y, x] = 255

    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.imshow('k', k)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result


if __name__ == '__main__':
    split_colors(cv2.imread('out/29/light.png'))
