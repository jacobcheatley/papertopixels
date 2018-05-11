import cv2

# CONSTANTS
MAX_DIMENSION = 1024 * 2
# This might want to be larger? Scaling might not even be needed


def resize(img, dimension=MAX_DIMENSION):
    height, width = img.shape[:2]
    if height > dimension or width > dimension:
        sf = min(dimension / height, dimension / width)
        return cv2.resize(img, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA)
    return img
