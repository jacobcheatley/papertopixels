import cv2
import numpy as np
import image_config

# CONSTANTS
MORPH = 3
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
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 4, 10, 120)
    # Edge detection
    edges = cv2.Canny(gray, 30, CANNY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Contours from the edges
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    highlight = gray.copy()

    rect = None

    # Find largest rectangle
    for cont in contours:
        arc_len = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)
        cv2.drawContours(highlight, [approx], -1, (255, 0, 0), 2)

        if len(approx) == 4:  # Make sure it's a quadrangle
            pts_src = np.array(approx, np.float32)

            if np.argmin(np.sum(pts_src.squeeze(), axis=1)) != 0:
                # If first point is not top left, must be top right
                pts_src = np.roll(pts_src, 3, axis=0)

            h, status = cv2.findHomography(pts_src, pts_dst)
            rect = cv2.warpPerspective(img, h, (int(WIDTH + MARGIN * 2), int(HEIGHT + MARGIN * 2)))
            break

    return closed, highlight, rect, 1.4142


def rectangle_preview(img):
    # Misc little debugging function to display results of rectangle selection
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
