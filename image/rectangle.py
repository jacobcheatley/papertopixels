import cv2
import numpy as np

# CONSTANTS
MORPH = 7
CANNY = 250

WIDTH = 512.0
HEIGHT = 512.0
MARGIN = 0.0

corners = np.array([[[MARGIN, MARGIN]],
                    [[MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, MARGIN]], ])

pts_dst = np.array(corners, np.float32)


def edges_highlight_rect(img):
    # Noise reduced greyscale image
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1, 10, 120)
    # Edge detection
    edges = cv2.Canny(gray, 10, CANNY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    _, contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect = None

    # Find a rectangle
    # TODO: Largest rectangle
    for cont in contours:
        arc_len = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

        if len(approx) == 4:
            pts_src = np.array(approx, np.float32)
            h, status = cv2.findHomography(pts_src, pts_dst)
            rect = cv2.warpPerspective(img, h, (int(WIDTH + MARGIN * 2), int(HEIGHT + MARGIN * 2)))
            cv2.drawContours(img, [approx], -1, (255, 0, 0), 2)
            break

    return edges, img, rect


def rectangle_preview(img):
    edges, highlight, rect = edges_highlight_rect(img)

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
