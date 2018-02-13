import cv2

# CONSTANTS
CLIP = 3.0
TILE = 16


def normalize_light(img):
    # Simple histogram equalization
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLIP, tileGridSize=(TILE, TILE))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
