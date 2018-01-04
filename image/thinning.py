from cv2 import ximgproc
import cv2
import numpy as np

CLOSE_SE = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


def _remove_stair(thinned: np.ndarray):
    # This will probably be slow because python loops
    # Based on remove_staircases from https://github.com/yati-sagade/zhang-suen-thinning/blob/master/zhangsuen.cpp
    # TODO: Profiling?
    points_y = []
    points_x = []
    rows, cols = np.where(thinned == 255)
    for i in range(2):
        for y, x in zip(rows, cols):
            # Center and 8 directional neighbours
            nw, n, ne = thinned[y - 1, x - 1:x + 2]
            w, c, e = thinned[y, x - 1:x + 2]
            sw, s, se = thinned[y + 1, x - 1:x + 2]

            # TODO: Make this code readable and understandable
            if i == 0:
                # North biased staircase removal
                if not (c and not (n and ((e and not ne and not sw and (not w or not s)) or (w and not nw and not se and (not e or not s))))):
                    points_y.append(y)
                    points_x.append(x)
            else:
                # South biased staircase removal
                if not (c and not (s and ((e and not se and not nw and (not w or not n)) or (w and not sw and not ne and (not e or not n))))):
                    points_y.append(y)
                    points_x.append(x)

            thinned[points_y, points_x] = 0
            points_y.clear()
            points_x.clear()


def thin_lines(img, destair=True):
    # Do a close open to fix up gaps
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, CLOSE_SE)

    thinned = ximgproc.thinning(img, thinningType=ximgproc.THINNING_ZHANGSUEN)

    if destair:
        _remove_stair(thinned)

    return thinned
