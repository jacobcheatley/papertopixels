import cv2
import numpy as np
import image_config

# DETECTION CONSTANTS
MORPH = 7
KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
SETS = [(10, 120, [70]),
        (10, 170, [5, 30, 120]),
        (30, 170, [30]),
        (30, 200, [70]),
        (30, 230, [5, 120]),
        (60, 200, [10]),
        (70, 230, [30]),
        ]
# Sets are MIN, MAX, [SIGMA]
D = 6

# WARPING
WIDTH = float(image_config.RESOLUTION)
HEIGHT = float(image_config.RESOLUTION)
MARGIN = 0.0

corners = np.array([[[MARGIN, MARGIN]],
                    [[MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, HEIGHT + MARGIN]],
                    [[WIDTH + MARGIN, MARGIN]], ])

pts_dst = np.array(corners, np.float32)


def try_find(img, MIN=30, MAX=250, SIG_C=10, SIG_S=120, gray=None):
    if gray is None:
        target_scale = 1
        gray = cv2.bilateralFilter(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), D, SIG_C, SIG_S)
    else:
        target_scale = max(img.shape) / max(gray.shape)

    # Edge detection
    edges = cv2.Canny(gray, MIN, MAX)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, KERNEL)
    # Contours from the edges
    _, contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if image_config.DETAIL_SAVE_IMAGES:
        cv2.imwrite(f'image/detail/edge-{MIN}-{MAX}.png', edges)
        cv2.imwrite(f'image/detail/closed-{MIN}-{MAX}.png', closed)
        cont_img = img.copy()
        cv2.drawContours(cont_img, np.int32(np.float32(contours) * target_scale), -1, (255, 0, 0), 5)
        cv2.imwrite(f'image/detail/contour-{MIN}-{MAX}.png', cont_img)

    rect = None
    area = 0

    # Find largest rectangle
    for cont in contours:
        arc_len = cv2.arcLength(cont, True)
        approx = cv2.approxPolyDP(cont, 0.1 * arc_len, True)

        if len(approx) == 4:  # Make sure it's a quadrangle
            pts_src = np.array(approx, np.float32)

            if np.argmin(np.sum(pts_src.squeeze(), axis=1)) != 0:
                # If first point is not top left, must be top right
                pts_src = np.roll(pts_src, 3, axis=0)

            if image_config.DETAIL_SAVE_IMAGES:
                highlight = img.copy()
                cv2.drawContours(highlight, [np.int32(approx * target_scale)], -1, (0, 255, 0), 10)
                cv2.imwrite(f'image/detail/highlight-{MIN}-{MAX}.png', highlight)

            h, status = cv2.findHomography(pts_src * target_scale, pts_dst)
            rect = cv2.warpPerspective(img, h, (int(WIDTH + MARGIN * 2), int(HEIGHT + MARGIN * 2)))
            area = cv2.contourArea(cont)
            break

    return rect, area


def best_rectangle(img, do_denoise=True):
    potentials = []

    if do_denoise:  # TVL1
        from .scaling import resize
        from .denoising import denoise

        gray = denoise(resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1024))

        for MIN, MAX, _ in SETS:
            rect, area = try_find(img, MIN=MIN, MAX=MAX, gray=gray)
            if rect is not None:
                potentials.append((rect, (area, MIN, MAX, 0)))
    else:  # Try different bilateral filter sigmas
        for MIN, MAX, SIGS in SETS:
            for SIG in SIGS:
                rect, area = try_find(img, MIN=MIN, MAX=MAX, SIG_C=SIG, SIG_S=SIG)
                if rect is not None:
                    potentials.append((rect, (area, MIN, MAX, SIG)))

    potentials = sorted(potentials, key=lambda d: d[1][0], reverse=True)

    return potentials[0][0] if len(potentials) > 0 else None
