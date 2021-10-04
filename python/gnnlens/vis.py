# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

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
