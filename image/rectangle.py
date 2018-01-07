import cv2
import numpy as np
from math import sqrt
import image_config

# CONSTANTS
MORPH = 7
CANNY = 250

WIDTH = float(image_config.RESOLUTION)
HEIGHT = float(image_config.RESOLUTION)
MARGIN = 0.0

corners = np.array([[[MARGIN, MARGIN]],
                    [[MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, MARGIN]], ])

pts_dst = np.array(corners, np.float32)


def edges_highlight_rect_ratio(img):
    # Noise reduced greyscale image
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 10, 120)
    # Edge detection
    edges = cv2.Canny(gray, 10, CANNY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = None
    ratio = 1.4142

    # Find largest rectangle
    # TODO: Rough backgrounds need different filter work
    max_area = 0

    for cont in contours:
        arc_len = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

        area = cv2.contourArea(cont)

        if area > max_area and len(approx) == 4:
            max_area = area

            pts_src = np.array(approx, np.float32)

            if np.argmin(np.sum(pts_src.squeeze(), axis=1)) != 0:
                # If first point is not top left, must be top right
                pts_src = np.roll(pts_src, 3, axis=0)

            h, status = cv2.findHomography(pts_src, pts_dst)
            rect = cv2.warpPerspective(img, h, (int(WIDTH + MARGIN * 2), int(HEIGHT + MARGIN * 2)))
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
            # ratio = approximate_ratio(pts_src, img.shape[1], img.shape[0])
            # TODO: Disabled for now, having problems with ratio calculations

    return edges, img, rect, ratio


def approximate_ratio(points, w, h):
    # https://stackoverflow.com/a/1222855
    # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.578.2050&rep=rep1&type=pdf
    # I love how mathematicians don't use clear variable names
    # s = 1 here, square pixels
    u0 = w / 2
    v0 = h / 2
    vert = (points - np.array([u0, v0])).squeeze()
    # Expected order: bl, br, tl, tr
    m = np.append(vert, [[1], [1], [1], [1]], axis=1)
    # Order must be tl, bl, br, tr coming from pts finding
    print(m)
    m2, m0, m1, m3 = m
    print('bl', m0, 'br', m1, 'tl', m2, 'tr', m3)

    k2 = np.dot(np.cross(m0, m3), m2) / np.dot(np.cross(m1, m3), m2)
    k3 = np.dot(np.cross(m0, m3), m1) / np.dot(np.cross(m2, m3), m1)

    # Special case for k2 ~= k3 ~= 1
    if abs(k2 - 1) < 0.01 and abs(k3 - 1) < 0.01:
        return 1 / sqrt(((m1[1] - m0[1]) ** 2 + (m1[0] - m0[0]) ** 2) /
                        ((m2[1] - m0[1]) ** 2 + (m2[0] - m0[0]) ** 2))

    n2 = k2 * m1 - m0
    n3 = k3 * m2 - m0

    f2 = -1 / (n2[2] * n3[2]) * (
            (n2[0] * n3[0] - (n2[0] * n3[2] + n2[2] * n3[0]) * u0 + n2[2] * n3[2] * (u0 ** 2)) +
            (n2[1] * n3[1] - (n2[1] * n3[2] + n2[2] * n3[1]) * v0 + n2[2] * n3[2] * (v0 ** 2))
        )

    f = sqrt(f2)

    a = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
    a_inv = np.linalg.inv(a)
    at_inv = np.linalg.inv(a.transpose())

    return sqrt(((n2 @ at_inv) @ (a_inv @ np.transpose(n2))) /
                ((n3 @ at_inv) @ (a_inv @ np.transpose(n3))))


def rectangle_preview(img):
    edges, highlight, rect, _ = edges_highlight_rect_ratio(img)

    cv2.namedWindow('edges', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('edges', edges)
    cv2.namedWindow('rgb', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('rgb', img)
    if rect is not None:
        cv2.namedWindow('out', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('out', rect)
    cv2.waitKey(0)


if __name__ == '__main__':
    img = cv2.imread('data/maze-scaled.jpg')
    rectangle_preview(img)
