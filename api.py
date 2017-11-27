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
    os.makedirs(f'{pre}/lines', exist_ok=True)

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

    # Lines
    b_lines = image.refine_lines(b)
    g_lines = image.refine_lines(g)
    r_lines = image.refine_lines(r)
    k_lines = image.refine_lines(k)
    cv2.imwrite(f'{pre}/lines/b.png', b_lines)
    cv2.imwrite(f'{pre}/lines/g.png', g_lines)
    cv2.imwrite(f'{pre}/lines/r.png', r_lines)
    cv2.imwrite(f'{pre}/lines/k.png', k_lines)

    _json_save(
        {
            'id': map_id,
            'original_size': img.shape[:2],
            'scale_size': scaled.shape[:2],
            'dim': list(rect.shape),
            'ratio': 1.4142,
            'lines': []
        }
    )

    return map_id


def get_next_free():
    return len(os.listdir('./maps'))
