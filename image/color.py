import numpy as np
import cv2

BLUE_MASK = (90, 150)
GREEN_MASK = (30, 90)
RED_L_MASK = (150, 179)
RED_U_MASK = (0, 30)

SAT = (50, 255)
VAL = (50, 255)


def create_masks(template):
    return np.array([template[0], SAT[0], VAL[0]]), np.array([template[1], SAT[1], VAL[1]])


B1, B2 = create_masks(BLUE_MASK)
G1, G2 = create_masks(GREEN_MASK)
RL1, RL2 = create_masks(RED_L_MASK)
RU1, RU2 = create_masks(RED_U_MASK)


def threshold(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    b = cv2.inRange(hsv, B1, B2)
    g = cv2.inRange(hsv, G1, G2)
    rl = cv2.inRange(hsv, RL1, RL2)
    ru = cv2.inRange(hsv, RU1, RU2)
    r = rl + ru

    cv2.imshow('b', b)
    cv2.imshow('g', g)
    cv2.imshow('r', r)
    cv2.waitKey(0)


if __name__ == '__main__':
    threshold(cv2.imread('out/1/rect.png'))
