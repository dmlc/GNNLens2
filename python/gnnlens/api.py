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

import logging

from flask import request, jsonify, Blueprint, current_app

from .db import *
api = Blueprint('api', __name__)

logger = logging.getLogger('gnn_vis')


######################
# API Starts here
######################
@api.route('/graphs', methods=['GET'])
def get_datasets():
    """Fetch the info of all datasets"""
    logdir = current_app.config['LOGDIR']
    dataset_list = getDatasetsList(logdir)
    if dataset_list is None:
        return jsonify({'success': False, "info": "Something goes wrong when fetching info of datasets." })
    else:
        return jsonify(dataset_list)

@api.route('/models', methods=['GET'])
def get_modelslist():
    """Fetch the info of models"""
    logdir = current_app.config['LOGDIR']
    dataset_id = request.args.get('dataset_id', None, type=int)
    if dataset_id is None:
        return jsonify({'success': False, "info": "Please provide dataset_id." })
    else:
        modellist = getModelsList(logdir, dataset_id)
        if modellist is None:
            return jsonify({'success': False, "info": "Something goes wrong when fetching info of models." })
        else:
            #return jsonify({'success': True, "graph_obj": graph_obj })
            return jsonify(modellist)
            
            
@api.route('/subgraphs', methods=['GET'])
def get_subgraphslist():
    """Fetch the info of subgraphs"""
    logdir = current_app.config['LOGDIR']
    dataset_id = request.args.get('dataset_id', None, type=int)
    if dataset_id is None:
        return jsonify({'success': False, "info": "Please provide dataset_id." })
    else:
        subgraphlist = getSubGraphsList(logdir, dataset_id)
        if subgraphlist is None:
            return jsonify({'success': False, "info": "Something goes wrong when fetching info of subgraphs." })
        else:
            #return jsonify({'success': True, "graph_obj": graph_obj })
            return jsonify(subgraphlist)            

@api.route('/graphinfo', methods=['GET'])
def get_graph_info():
    """Fetch the info of specific graph"""
    logdir = current_app.config['LOGDIR']
    dataset_id = request.args.get('dataset_id', None, type=int)
    if dataset_id is None:
        return jsonify({'success': False, "info": "Please provide dataset_id." })
    else:
        graph_obj = getGraphInfo(logdir, dataset_id)
        if graph_obj is None:
            return jsonify({'success': False, "info": "Something goes wrong when fetching info of graphs." })
        else:
            #return jsonify({'success': True, "graph_obj": graph_obj })
            return jsonify(graph_obj)
        
@api.route('/modelinfo', methods=['GET'])
def get_model_info():
    """Fetch the info of specific graph"""
    logdir = current_app.config['LOGDIR']
    dataset_id = request.args.get('dataset_id', None, type=int)
    model_id = request.args.get('model_id', None, type=int)
    if dataset_id is None or model_id is None:
        return jsonify({'success': False, "info": "Please provide dataset_id and model_id." })
    else:
        model_obj = getModelInfo(logdir, dataset_id, model_id)
        if model_obj is None:
            return jsonify({'success': False, "info": "Something goes wrong when fetching info of models." })
        else:
            #return jsonify({'success': True, "graph_obj": graph_obj })
            return jsonify(model_obj)

@api.route('/subgraphinfo', methods=['GET'])
def get_subgraph_info():
    """Fetch the info of specific graph"""
    logdir = current_app.config['LOGDIR']
    dataset_id = request.args.get('dataset_id', None, type=int)
    subgraph_id = request.args.get('subgraph_id', None, type=int)
    if dataset_id is None or subgraph_id is None:
        return jsonify({'success': False, "info": "Please provide dataset_id and subgraph_id." })
    else:
        subgraph_obj = getSubGraphInfo(logdir, dataset_id, subgraph_id)
        if subgraph_obj is None:
            return jsonify({'success': False, "info": "Something goes wrong when fetching info of subgraphs." })
        else:
            #return jsonify({'success': True, "graph_obj": graph_obj })
            return jsonify(subgraph_obj)