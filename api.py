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
    map_id = get_next_free()  # TODO: support concurrency properly

    # Get image into memory to be worked with in CV
    in_memory = BytesIO()
    file.save(in_memory)
    data = np.fromstring(in_memory.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 1)

    # Process the image in stages
    rect = image.find_rectangle(img)
    lines = image.find_lines(rect)  # TODO: Make this actually functional
    _json_save({'id': map_id, 'dim': list(rect.shape)})

    return map_id


def get_next_free():
    return len(os.listdir('./maps'))
