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

import json
import os
from pathlib import Path

def read_json(file_name):
    with open(file_name, "r") as f:
        obj = json.load(f)
    return obj

def getDatasetsList(logdir):
    filename = Path(logdir) / 'dataset_list.json'
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        datasets_released_list = read_json(filename)
        return datasets_released_list

def getModelsList(logdir, dataset_id):
    filename = Path(logdir) / '{}/model_list.json'.format(dataset_id)
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        models_released_list = read_json(filename)
        return models_released_list


def getSubGraphsList(logdir, dataset_id):
    filename = Path(logdir) / '{}/subgraph_list.json'.format(dataset_id)
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        subgraphs_released_list = read_json(filename)
        return subgraphs_released_list


def getGraphInfo(logdir, dataset_id):
    filename = Path(logdir) / '{}/graph.json'.format(dataset_id)
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        graph_pkg = read_json(filename)
        return graph_pkg

def getModelInfo(logdir, dataset_id, model_id):
    filename = Path(logdir) / '{}/model_{}.json'.format(dataset_id, model_id)
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        model_pkg = read_json(filename)
        return model_pkg

def getSubGraphInfo(logdir, dataset_id, subgraph_id):
    filename = Path(logdir) / '{}/subgraph_{}.json'.format(dataset_id, subgraph_id)
    if not os.path.isfile(filename):
        print("{} is not existed".format(filename))
        return None
    else:
        subgraph_pkg = read_json(filename)
        return subgraph_pkg