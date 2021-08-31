import os
import socket
import sys
CURRENT_ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, '..')))
sys.path.append(SERVER_ROOT)

import argparse
try:
    import simplejson as json
except ImportError:
    import json
from flask import Flask
from flask_cors import CORS
from mini_server.api import api
from mini_server.vis import vis
#from flask_compress import Compress
class Server_Config(object):
    def __init__(self, config):
        self.FRONT_ROOT = os.path.join(CURRENT_ROOT, './visbuild')
        self.STATIC_FOLDER = os.path.join(CURRENT_ROOT, './visbuild/static')
        self.LOGDIR = config["logdir"]
        
def create_app(config=None):
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    CORS(app)
    #Compress(app)
    @app.route('/hello')
    def hello():
        return 'Hello, World!'
    server_config = Server_Config(config)
    print("Server config:", vars(server_config))
    app.config.from_object(server_config)
    app.register_blueprint(api, url_prefix='/api')
    app.register_blueprint(vis, url_prefix='/')
    return app


def add_arguments_server(parser):

    # API flags
    parser.add_argument('--host', default='0.0.0.0', help='Port in which to run the API')
    parser.add_argument('--port', default=7777, type=int, help='Port in which to run the API')
    parser.add_argument('--logdir', default='./logs', help='Log directory')
    parser.add_argument('--debug', action="store_const", default=False, const=True,
                        help='If true, run Flask in debug mode')


def start_server():
    parser = argparse.ArgumentParser()
    add_arguments_server(parser)
    _args = parser.parse_args()

    if _args.debug:
        os.environ['FLASK_ENV'] = 'development'

    # Check if the port is already in use.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        result = s.connect_ex((_args.host, _args.port))
        if result == 0:
            raise OSError('Port {:d} already in use. You can specify a '
                          'different port with gnnvis --port xxxx'.format(_args.port))

    app = create_app(vars(_args))

    app.run(
        debug=_args.debug,
        host=_args.host,
        port=int(_args.port)
    )


if __name__ == '__main__':
    start_server()
