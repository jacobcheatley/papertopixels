import cv2
import numpy as np
import image_config

# DETECTION CONSTANTS
MORPH = 3
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
SETS = [(10, 170, 30), (30, 170, 30), (30, 200, 70), (30, 230, 5), (30, 230, 120)]
# Sets are MIN, MAX, SIGMA
D = 4

# WARPING
WIDTH = float(image_config.RESOLUTION)
HEIGHT = float(image_config.RESOLUTION)
MARGIN = 0.0

corners = np.array([[[MARGIN, MARGIN]],
                    [[MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, MARGIN]], ])

pts_dst = np.array(corners, np.float32)


def try_find(img, MIN=30, MAX=250, SIG_C=10, SIG_S=120):
    # Noise reduced greyscale image
    gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), D, SIG_C, SIG_S)
    # Edge detection
    edges = cv2.Canny(gray, MIN, MAX)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, KERNEL)
    # Contours from the edges
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    highlight = gray.copy()

    rect = None
    area = 0

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
            area = cv2.contourArea(cont)
            break

    return rect, area


def best_rectangle(img):
    potentials = []

    for MIN, MAX, SIG in SETS:
        rect, area = try_find(img, MIN=MIN, MAX=MAX, SIG_C=SIG, SIG_S=SIG)
        if rect is not None:
            potentials.append((rect, (area, MIN, MAX, SIG)))

    potentials = sorted(potentials, key=lambda d: d[1][0], reverse=True)

    return potentials[0][0] if len(potentials) > 0 else None


if __name__ == '__main__':
    from image.scaling import resize

    for i in range(1, 16):
        test_img = cv2.imread(f'/home/jcheatley/PycharmProjects/papertopixels/image/data/test/test{i}.jpg')
        test_img = resize(test_img)
        cv2.imshow('BEST', best_rectangle(test_img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
