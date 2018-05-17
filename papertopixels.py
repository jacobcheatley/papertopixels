from flask import Flask, request, redirect, render_template, url_for, flash, send_from_directory
import api
import app_config
import os
import json

app = Flask(__name__)
app.config.from_object(app_config.Config)
app.secret_key = 'test'
app.config['SESSION_TYPE'] = 'filesystem'
maps_folder = os.path.join(app.root_path, 'maps')
thumb_folder = os.path.join(app.root_path, 'thumb')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part.')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file.')
            return redirect(request.url)

        if file:
            map_id = api.process_image(file)

            if map_id is None and map_id != -1:
                flash('Timed out.')
                return redirect(request.url)
            elif map_id == -1:
                flash('Something strange happened. Wrong filetype?')
                return redirect(request.url)
            else:
                return redirect(url_for('view_map', map_id=map_id))
    else:
        return render_template('upload.html')


# TODO: remove and use apache file serving later
@app.route('/map/<int:map_id>', methods=['GET'])
def get_map(map_id: int):
    return send_from_directory(maps_folder, f'{map_id}.json')


@app.route('/thumb/<int:map_id>', methods=['GET'])
def get_thumbnail(map_id: int):
    return send_from_directory(thumb_folder, f'{map_id}.png')


@app.route('/maps', methods=['GET'])
def get_map_list():
    return json.dumps({'maps': api.get_all_maps()})


@app.route('/view', methods=['GET'], defaults={'map_id': 1})
@app.route('/view/<int:map_id>')
def view_map(map_id: int):
    return render_template('view.html', map_id=map_id)


if __name__ == '__main__':
    app.run(host='0.0.0.0')
