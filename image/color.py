import numpy as np
import cv2
import image_config

# CONSTANTS
BLUE_MASK = (90, 149)
GREEN_MASK = (30, 89)
RED_L_MASK = (150, 180)
RED_U_MASK = (0, 30)
BLACK_M_VALUE = 100

SAT = (50, 255)
VAL = (50, 255)

EDGE = 6

ERODE = 3
ERODE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (ERODE, ERODE))
DILATE = 5
DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (DILATE, DILATE))

edge_mask = np.zeros((image_config.RESOLUTION, image_config.RESOLUTION, 1), np.uint8)
cv2.rectangle(edge_mask, (EDGE-1, EDGE-1), (image_config.RESOLUTION-1-EDGE, image_config.RESOLUTION-1-EDGE), 255, thickness=cv2.FILLED)


def create_masks(template):
    return np.array([template[0], SAT[0], VAL[0]]), np.array([template[1], SAT[1], VAL[1]])


B1, B2 = create_masks(BLUE_MASK)
G1, G2 = create_masks(GREEN_MASK)
RL1, RL2 = create_masks(RED_L_MASK)
RU1, RU2 = create_masks(RED_U_MASK)
K1, K2 = np.array([0, 0, 0]), np.array([180, 255, BLACK_M_VALUE])


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

    # Remove edge noise (from slightly off edge detection)
    b = cv2.bitwise_and(b, edge_mask)
    g = cv2.bitwise_and(g, edge_mask)
    r = cv2.bitwise_and(r, edge_mask)
    k = cv2.bitwise_and(k, edge_mask)

    # # Do small erode and larger dilate
    # def noise_and_close(channel):
    #     return cv2.morphologyEx(cv2.morphologyEx(channel, cv2.MORPH_ERODE, ERODE_KERNEL), cv2.MORPH_DILATE, DILATE_KERNEL)
    #
    # b = noise_and_close(b)
    # g = noise_and_close(g)
    # r = noise_and_close(r)
    # k = noise_and_close(k)
    b = cv2.morphologyEx(cv2.medianBlur(b, 3), cv2.MORPH_DILATE, DILATE_KERNEL)
    g = cv2.morphologyEx(cv2.medianBlur(g, 3), cv2.MORPH_DILATE, DILATE_KERNEL)
    r = cv2.morphologyEx(cv2.medianBlur(r, 3), cv2.MORPH_DILATE, DILATE_KERNEL)
    k = cv2.morphologyEx(cv2.medianBlur(k, 3), cv2.MORPH_DILATE, DILATE_KERNEL)

    return b, g, r, k
