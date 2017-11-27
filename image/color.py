import numpy as np
import cv2

# CONSTANTS
BLUE_MASK = (90, 149)
GREEN_MASK = (30, 89)
RED_L_MASK = (150, 180)
RED_U_MASK = (0, 30)

SAT = (50, 255)
VAL = (50, 255)

EDGE = 4

edge_mask = np.zeros((512, 512, 1), np.uint8)
cv2.rectangle(edge_mask, (EDGE-1, EDGE-1), (511-EDGE, 511-EDGE), 255, thickness=cv2.FILLED)


def create_masks(template):
    return np.array([template[0], SAT[0], VAL[0]]), np.array([template[1], SAT[1], VAL[1]])


B1, B2 = create_masks(BLUE_MASK)
G1, G2 = create_masks(GREEN_MASK)
RL1, RL2 = create_masks(RED_L_MASK)
RU1, RU2 = create_masks(RED_U_MASK)
K1, K2 = np.array([0, 0, 0]), np.array([180, 255, 127])


def split_colors(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Colors
    b = cv2.inRange(hsv, B1, B2)
    g = cv2.inRange(hsv, G1, G2)
    rl = cv2.inRange(hsv, RL1, RL2)
    ru = cv2.inRange(hsv, RU1, RU2)
    r = rl + ru

    # Get all things which are NOT rgb
    combined = cv2.bitwise_not(cv2.bitwise_or(cv2.bitwise_or(b, g), r))

    # Black
    k = cv2.inRange(hsv, K1, K2)
    # Remove things which were previously classified as RGB
    k = cv2.bitwise_and(k, combined)

    # Remove edge noise
    b = cv2.bitwise_and(b, edge_mask)
    g = cv2.bitwise_and(g, edge_mask)
    r = cv2.bitwise_and(r, edge_mask)
    k = cv2.bitwise_and(k, edge_mask)

    # Remove some noise from black - using hit or miss
    # https://stackoverflow.com/questions/46143800/removing-isolated-pixels-using-opencv
    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)
    hom1 = cv2.morphologyEx(k, cv2.MORPH_ERODE, kernel1)
    hom2 = cv2.morphologyEx(cv2.bitwise_not(k), cv2.MORPH_ERODE, kernel2)
    hom = cv2.bitwise_not(cv2.bitwise_and(hom1, hom2))
    k = cv2.bitwise_and(k, hom)

    return b, g, r, k
