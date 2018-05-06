import cv2

# CONSTANTS
MAX_DIMENSION = 1024 * 2
# This might want to be larger? Scaling might not even be needed


def resize(img):
    height, width = img.shape[:2]
    if height > MAX_DIMENSION or width > MAX_DIMENSION:
        sf = min(MAX_DIMENSION / height, MAX_DIMENSION / width)
        return cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    return img
