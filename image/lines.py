from cv2 import ximgproc


def thin_lines(img):
    return ximgproc.thinning(img, thinningType=ximgproc.THINNING_ZHANGSUEN)
