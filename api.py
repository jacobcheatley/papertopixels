import werkzeug.datastructures
from io import BytesIO
import numpy as np
import cv2
import image
import json
import os


def get_map(map_id: int):
    return {'id': map_id}


def _json_save(dct: dict):
    with open(f'maps/{dct["id"]}.json', 'w') as out:
        json.dump(dct, out)


def process_image(file: werkzeug.datastructures.FileStorage):
    print('Processing image')

    # Set up new folder for intermediate images
    map_id = get_next_free()  # TODO: support concurrency properly
    pre = f'image/out/{map_id}'
    os.makedirs(f'{pre}/colors', exist_ok=True)
    os.makedirs(f'{pre}/lines/contours', exist_ok=True)

    # Get image into memory to be worked with in CV
    in_memory = BytesIO()
    file.save(in_memory)
    data = np.fromstring(in_memory.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)

    # Process the image in stages
    # Scale image to max size
    scaled = image.resize(img)
    cv2.imwrite(f'{pre}/scaled.png', scaled)

    # Find paper rectangle
    edges, highlight, rect = image.edges_highlight_rect(scaled)
    cv2.imwrite(f'{pre}/edges.png', edges)  # *
    cv2.imwrite(f'{pre}/highlight.png', highlight)  # *
    cv2.imwrite(f'{pre}/rect.png', rect)

    # Lighting correction
    light = image.normalize_light(rect)
    cv2.imwrite(f'{pre}/light.png', light)

    # Color thresholding
    b, g, r, k = image.split_colors(light)
    cv2.imwrite(f'{pre}/colors/b.png', b)
    cv2.imwrite(f'{pre}/colors/g.png', g)
    cv2.imwrite(f'{pre}/colors/r.png', r)
    cv2.imwrite(f'{pre}/colors/k.png', k)

    # Thinning
    b_thin = image.thin_lines(b)
    g_thin = image.thin_lines(g)
    r_thin = image.thin_lines(r)
    k_thin = image.thin_lines(k)
    cv2.imwrite(f'{pre}/lines/b.png', b_thin)
    cv2.imwrite(f'{pre}/lines/g.png', g_thin)
    cv2.imwrite(f'{pre}/lines/r.png', r_thin)
    cv2.imwrite(f'{pre}/lines/k.png', k_thin)

    # Connectivity
    b_contours = image.get_contours(b_thin)
    g_contours = image.get_contours(g_thin)
    r_contours = image.get_contours(r_thin)
    k_contours = image.get_contours(k_thin)
    cv2.imwrite(f'{pre}/lines/contours/b.png', image.contour_image(b_thin, b_contours))
    cv2.imwrite(f'{pre}/lines/contours/g.png', image.contour_image(g_thin, g_contours))
    cv2.imwrite(f'{pre}/lines/contours/r.png', image.contour_image(r_thin, r_contours))
    cv2.imwrite(f'{pre}/lines/contours/k.png', image.contour_image(k_thin, k_contours))

    lines = list(image.get_all_lines(
        (b_thin, 'b'),
        (g_thin, 'g'),
        (r_thin, 'r'),
        (k_thin, 'k'),
    ))

    _json_save(
        {
            'id': map_id,
            'ratio': 1.4142,
            'lines': lines
        }
    )

    return map_id


def get_next_free():
    return len(os.listdir('./maps'))
