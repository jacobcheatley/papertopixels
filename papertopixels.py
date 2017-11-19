from flask import Flask, request, redirect, render_template, url_for, flash, send_from_directory
import api
import config
import os

app = Flask(__name__)
app.config.from_object(config.Config)
maps_folder = os.path.join(app.root_path, 'maps')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file:
            map_id = api.process_image(file)
            return redirect(url_for('get_map', map_id=map_id))
    else:
        return render_template('upload.html')


# TODO: remove and use apache file serving later
@app.route('/map/<map_id>', methods=['GET'])
def get_map(map_id: int):
    return send_from_directory(maps_folder, f'{map_id}.json')


if __name__ == '__main__':
    app.secret_key = 'test'
    app.config['SESSION_TYPE'] = 'filesystem'

    app.run()
