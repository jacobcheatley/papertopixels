import werkzeug.datastructures
from io import BytesIO
import numpy as np
import cv2
import image
import json
import os
import image_config
import time
import multiprocessing

path = os.path.dirname(os.path.realpath(__file__))
MAPS_FOLDER = os.path.join(path, 'maps')
THUMB_FOLDER = os.path.join(path, 'thumb')


def _json_save(dct: dict):
    with open(f'{MAPS_FOLDER}/{dct["id"]}.json', 'w') as out:
        json.dump(dct, out)


def print_time(fmt, start_time):
    print(fmt.format(time.time() - start_time))
    return time.time()


def _process_image(file: werkzeug.datastructures.FileStorage, output: multiprocessing.Value):
    # Resizing -> Rectangle extraction -> Lighting correction -> Foreground detection and masking
    # -> Thinning -> Segmentation -> Segment colour detection -> Joining up same colour segments appropriately
    try:
        if image_config.PRINT_TIMES:
            print('Processing image')

        # Set up new folder for intermediate images
        map_id = get_next_free()

        if image_config.SAVE_IMAGE:
            pre = f'image/out/{map_id}-{time.time()}'
            os.makedirs(f'{pre}', exist_ok=True)

        # Get image into memory to be worked with in CV
        in_memory = BytesIO()
        file.save(in_memory)
        data = np.fromstring(in_memory.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(data, 1)

        # Get start time for time debug printing
        process_start = time.time()
        start_time = process_start

        # Process the image in stages
        # Scale image to max size
        scaled = image.resize(img)

        if image_config.SAVE_IMAGE:
            cv2.imwrite(f'{pre}/scaled.png', scaled)
        if image_config.PRINT_TIMES:
            process_start = print_time('Resize {}', process_start)

        # Find paper rectangle
        edges, highlight, rect, ratio = image.edges_highlight_rect_ratio(scaled)

        if image_config.SAVE_IMAGE:
            cv2.imwrite(f'{pre}/edges.png', edges)
            cv2.imwrite(f'{pre}/highlight.png', highlight)
            cv2.imwrite(f'{pre}/rect.png', rect)
        if image_config.PRINT_TIMES:
            process_start = print_time('Rectangle {}', process_start)

        # Lighting correction
        light = image.normalize_light(rect)

        if image_config.SAVE_IMAGE:
            cv2.imwrite(f'{pre}/light.png', light)
        if image_config.PRINT_TIMES:
            process_start = print_time('Lighting {}', process_start)

        # Foreground and background
        mask = image.find_mask(light)

        if image_config.SAVE_IMAGE:
            cv2.imwrite(f'{pre}/mask.png', mask)
        if image_config.PRINT_TIMES:
            process_start = print_time('Mask {}', process_start)

        # Thinning
        thinned = image.thin_lines(mask)

        if image_config.SAVE_IMAGE:
            cv2.imwrite(f'{pre}/thinned.png', thinned)
        if image_config.PRINT_TIMES:
            process_start = print_time('Thinning {}', process_start)

        # Lines
        line_finder = image.LineFinder(light, thinned)
        lines = list(line_finder.get_all_lines())

        if image_config.PRINT_TIMES:
            process_start = print_time('Lines {}', process_start)

        map_data = {'id': map_id, 'ratio': ratio, 'resolution': image_config.RESOLUTION, 'lines': lines}

        thumb = image.generate_thumb(map_data)
        cv2.imwrite(f'{THUMB_FOLDER}/{map_id}.png', thumb)

        _json_save(map_data)

        if image_config.PRINT_TIMES:
            print_time('Saved {}', process_start)
            print_time('Total {}', start_time)

        output.value = map_id
        return
    except Exception as e:
        print(e)
        output.value = -1
        return


def process_image(file: werkzeug.datastructures.FileStorage):
    # Timeout logic
    map_id = multiprocessing.Value('i')
    p = multiprocessing.Process(target=_process_image, name="Process", args=(file, map_id,))
    p.start()
    p.join(image_config.TIMEOUT)

    if p.is_alive():  # Hasn't terminated, likely stuck
        p.terminate()
        return None
    else:
        return map_id.value


def get_next_free():
    # TODO: Concurrency support
    return len(os.listdir(MAPS_FOLDER))


def get_all_maps():
    # Give back an array of all map ids {'maps': [0,1,...]}
    name_stat = [(fn, os.stat(os.path.join(MAPS_FOLDER, fn))) for fn in os.listdir(MAPS_FOLDER)]
    ints = [int(ns[0].split('.')[0]) for ns in sorted(name_stat, key=lambda ns: ns[1].st_ctime, reverse=True)]

    return json.dumps({'maps': ints})
