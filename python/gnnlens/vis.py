from flask import Blueprint, send_from_directory, safe_join, current_app

vis = Blueprint('vis', __name__)


@vis.route('/static/js/<path:path>')
def send_js(path):
    return send_from_directory(safe_join(current_app.config['STATIC_FOLDER'], 'js'), path)


@vis.route('/static/css/<path:path>')
def send_css(path):
    return send_from_directory(safe_join(current_app.config['STATIC_FOLDER'], 'css'), path)


@vis.route('/static/media/<path:path>')
def send_media(path):
    return send_from_directory(safe_join(current_app.config['STATIC_FOLDER'], 'media'), path)


@vis.route('/')
def index():
    print("#index")
    return send_from_directory(current_app.config['FRONT_ROOT'], 'index.html')


@vis.route('/<string:model>', methods=['GET'])
def send_index(model):
    if model == 'service-worker.js':
        return send_from_directory(current_app.config['FRONT_ROOT'], 'service-worker.js')
    if model == 'favicon.ico':
        return send_from_directory(current_app.config['FRONT_ROOT'], 'favicon.ico')
    if model == 'index.html':
        return send_from_directory(current_app.config['FRONT_ROOT'], 'index.html')
    if model == 'manifest.json':
        return send_from_directory(current_app.config['FRONT_ROOT'], 'manifest.json')
