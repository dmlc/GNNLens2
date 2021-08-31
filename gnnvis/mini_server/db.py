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