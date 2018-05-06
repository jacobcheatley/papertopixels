import cv2
import numpy as np

# CONSTANTS
CLIP = 3.0
TILE = 16

KERNEL = (149, 149)
# KERNEL = (3, 3)

# def normalize_light(img):
#     # Simple histogram equalization
#     lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def normalize_light(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    lpf = cv2.GaussianBlur(v, KERNEL, 0)
    mean = np.ones(v.shape) * np.mean(lpf).astype(int)
    newv = v - lpf + mean
    hsv[:, :, 2] = newv
    # TEST
    print('ALL MEAN', np.mean(newv))
    # TEST
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)#, v, lpf, newv
#
#
# if __name__ == '__main__':
#     img = cv2.imread('/home/jcheatley/PycharmProjects/papertopixels/image/out/15-1524693137.4076707/rect.png')
#     light, v, lpf, newv = normalize_light(img)
#     cv2.imshow('Light', light)
#     # cv2.imshow('V', v)
#     cv2.imshow('LPF', lpf)
#     cv2.imshow('NewV', newv)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def normalize_light(img):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     v = hsv[:, :, 2]
#     newv = cv2.morphologyEx(v, cv2.MORPH_BLACKHAT, KERNEL)
#     hsv[:, :, 2] = newv
#     return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), newv
#
#
# if __name__ == '__main__':
#     img = cv2.imread('/home/jcheatley/PycharmProjects/papertopixels/image/out/15-1524693137.4076707/rect.png')
#     light, newv = normalize_light(img)
#     cv2.imshow('light', light)
#     cv2.imshow('newv', newv)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
