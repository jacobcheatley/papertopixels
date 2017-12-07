import cv2
import numpy as np

HEIGHT = 256


def generate_thumb(map_data):
    width = int(HEIGHT * map_data['ratio'])
    res = map_data['resolution']

    img = np.ones((HEIGHT, width, 3), np.uint8) * 255

    def to_pix(point):
        return int(width * point['x'] / res), int(HEIGHT * point['y'] / res)

    def get_color(col):
        if col == 'r':
            return 0, 0, 255
        elif col == 'g':
            return 0, 255, 0
        elif col == 'b':
            return 255, 0, 0
        else:
            return 0, 0, 0

    for line in map_data['lines']:
        points = list(map(to_pix, line['points']))
        color = get_color(line['color'])

        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i + 1], color, 2)

        if line['closed']:
            cv2.line(img, points[-1], points[0], color, 2)

    return img
