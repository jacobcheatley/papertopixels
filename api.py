import werkzeug.datastructures
from io import BytesIO
import numpy as np
import cv2
import image
import json
import os
import image_config

path = os.path.dirname(os.path.realpath(__file__))
MAPS_FOLDER = os.path.join(path, 'maps')
THUMB_FOLDER = os.path.join(path, 'thumb')


def _json_save(dct: dict):
    with open(f'{MAPS_FOLDER}/{dct["id"]}.json', 'w') as out:
        json.dump(dct, out)


def process_image(file: werkzeug.datastructures.FileStorage):
    print('Processing image')

    # Set up new folder for intermediate images
    map_id = get_next_free()  # TODO: support concurrency properly
    if image_config.SAVE_IMAGE:
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
    if image_config.SAVE_IMAGE:
        cv2.imwrite(f'{pre}/scaled.png', scaled)

    # Find paper rectangle
    edges, highlight, rect, ratio = image.edges_highlight_rect_ratio(scaled)
    if image_config.SAVE_IMAGE:
        cv2.imwrite(f'{pre}/edges.png', edges)  # *
        cv2.imwrite(f'{pre}/highlight.png', highlight)  # *
        cv2.imwrite(f'{pre}/rect.png', rect)

    # Lighting correction
    light = image.normalize_light(rect)
    if image_config.SAVE_IMAGE:
        cv2.imwrite(f'{pre}/light.png', light)

    # Color thresholding
    b, g, r, k = image.split_colors(light)
    if image_config.SAVE_IMAGE:
        cv2.imwrite(f'{pre}/colors/b.png', b)
        cv2.imwrite(f'{pre}/colors/g.png', g)
        cv2.imwrite(f'{pre}/colors/r.png', r)
        cv2.imwrite(f'{pre}/colors/k.png', k)

    # Thinning
    b_thin = image.thin_lines(b)
    g_thin = image.thin_lines(g)
    r_thin = image.thin_lines(r)
    k_thin = image.thin_lines(k)
    if image_config.SAVE_IMAGE:
        cv2.imwrite(f'{pre}/lines/b.png', b_thin)
        cv2.imwrite(f'{pre}/lines/g.png', g_thin)
        cv2.imwrite(f'{pre}/lines/r.png', r_thin)
        cv2.imwrite(f'{pre}/lines/k.png', k_thin)

    lines = list(image.get_all_lines(
        (b_thin, 'b'),
        (g_thin, 'g'),
        (r_thin, 'r'),
        (k_thin, 'k'),
    ))

    map_data = {'id': map_id, 'ratio': ratio, 'resolution': image_config.RESOLUTION, 'lines': lines}

    thumb = image.generate_thumb(map_data)
    cv2.imwrite(f'{THUMB_FOLDER}/{map_id}.png', thumb)

    _json_save(map_data)

    return map_id


def get_next_free():
    return len(os.listdir(MAPS_FOLDER))


def get_all_maps():
    name_stat = [(fn, os.stat(os.path.join(MAPS_FOLDER, fn))) for fn in os.listdir(MAPS_FOLDER)]
    ints = [int(ns[0].split('.')[0]) for ns in sorted(name_stat, key=lambda ns: ns[1].st_ctime, reverse=True)]

    return json.dumps({'maps': ints})
