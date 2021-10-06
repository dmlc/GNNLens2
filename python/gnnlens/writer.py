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

import dgl.backend as F
import json
import numpy as np
import os
from dgl import DGLError

__all__ = ['Writer']

class Writer():
    """
    Parameters
    ----------
    logdir: str
        Path to create a new directory for dumping data files, which can 
        be either a relative path or an absolute path.
    """
    def __init__(self, logdir):
        os.makedirs(logdir)
        self.logdir = logdir
        self.graph_names = []
        self.graph_data = dict()

    def _get_graph_logdir(self, name):
        """Get logdir for the graph.
        
        Parameters
        ----------
        name : str
            Name of the graph.
        """
        return os.path.join(self.logdir, str(self.graph_data[name]['id']))
    
    def add_graph(self, name, graph, nlabels=None, num_nlabel_types=None, eweights=None):
        """Add data for a graph.

        Parameters
        ----------
        name : str
            Name of the graph.
        graph : DGLGraph
            A homogeneous graph.
        nlabels : Tensor of integers, optional
            Node labels. The tensor can be reshaped as (N,) where N is the number of nodes.
            Each node should be associated with one label only.
        num_nlabel_types : int, optional
            Number of node label types. If not provided and nlabels is provided,
            this will be inferred then. num_nlabel_types should be no greater than 10.
        eweights : dict[str, tensor]
            Edge weights. The keys are the eweight names, e.g. confidence. The values are the
            tensors of edge weights. The tensors can be reshaped as (E,) where E is the number
            of edges.
        """
        if name in self.graph_names:
            raise DGLError('Graph name {} has already been used.'.format(name))

        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        
        srcs, dsts = graph.edges()
        srcs = F.asnumpy(srcs).tolist()
        dsts = F.asnumpy(dsts).tolist()

        # Handle nlabels
        if nlabels is None:
            nlabels = []
        else:
            nlabels = F.asnumpy(nlabels)
            try:
                nlabels = np.reshape(nlabels, (num_nodes,))
            except:
                raise DGLError('Node labels should be able to be reshaped as (num_nodes,)')
            if num_nlabel_types is None:
                num_nlabel_types = int(nlabels.max()) + 1
            assert num_nlabel_types <= 10, \
                'Expect num_nlabel_types to be no greater than 10, got {:d}'.format(
                    num_nlabel_types)
            nlabels = nlabels.tolist()
        
        # Handle eweights
        if eweights is None:
            eweights = dict()
        else:
            for ew_name, ew_values in eweights.items():
                ew_values = F.asnumpy(ew_values)
                try:
                    ew_values = np.reshape(ew_values, (num_edges,))
                except:
                    raise DGLError('Edge weights should be able to be reshaped as (num_edges,)')
                eweights[ew_name] = ew_values.tolist()

        graph_obj = {
            "name": name,
            "srcs": srcs,
            "dsts": dsts,
            "num_nodes": num_nodes,
            "nlabels": nlabels,
            "num_nlabel_types": num_nlabel_types,
            "eweights": eweights
        }
        data_obj = {
            "graph_obj": graph_obj,
            "success": True
        }

        graph_id = len(self.graph_names) + 1
        graph_logdir = os.path.join(self.logdir, str(graph_id))
        os.makedirs(graph_logdir)

        # Dump graph data file
        with open(graph_logdir + '/graph.json', 'w') as f:
            json.dump(data_obj, f)

        # Register graph name
        self.graph_names.append(name)
        self.graph_data[name] = {
            'id': graph_id,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'model_list': [],
            'subgraph_list': [],
            'subgraphs': dict()
        }

    def add_model(self, graph_name, model_name, nlabels, eweights=None):
        """Add data for a model
        
        Parameters
        ----------
        graph_name : str
            Name of the graph.
        model_name : str
            Nmae of the model.
        nlabels : Tensor of integers, optional
            Node labels. The tensor can be reshaped as (N,) where N is the number of nodes.
            Each node should be associated with one label only.
        eweights : dict[str, tensor]
            Edge weights. The keys are the eweight names, e.g. confidence. The values are the
            tensors of edge weights. The tensors can be reshaped as (E,) where E is the number
            of edges. The edge weights should be in range [0, 1].
        """
        assert graph_name in self.graph_names, \
            'Expect add_graph to be called first for graph {}'.format(graph_name)

        graph_logdir = self._get_graph_logdir(graph_name)
        
        # Handle nlabels
        nlabels = F.asnumpy(nlabels)
        num_nodes = self.graph_data[graph_name]['num_nodes']
        try:
            nlabels = np.reshape(nlabels, (num_nodes,))
        except:
            raise DGLError('Node labels should be able to be reshaped as (num_nodes,)')
        nlabels = nlabels.tolist()

        # Handle eweights
        if eweights is None:
            eweights = dict()
        else:
            num_edges = self.graph_data[graph_name]['num_edges']
            for ew_name, ew_values in eweights.items():
                ew_values = F.asnumpy(ew_values)
                try:
                    ew_values = np.reshape(ew_values, (num_edges,))
                except:
                    raise DGLError('Edge weights should be able to be reshaped as (num_edges,)')
                eweights[ew_name] = ew_values.tolist()
        
        # Dump model data file
        num_models = len(self.graph_data[graph_name]['model_list']) + 1
        model_obj = {
            "name": model_name,
            "nlabels": nlabels,
            "eweights": eweights
        }
        with open(graph_logdir + '/model_{}.json'.format(num_models), 'w') as f:
            json.dump({"model_obj": model_obj, "success": True}, f)

        # Register the model
        self.graph_data[graph_name]['model_list'].append(model_name)

    def add_subgraph(self, graph_name, subgraph_name, node_id, subgraph_nids, subgraph_eids, 
                     subgraph_nweights=None, subgraph_eweights=None):
        """Add data for a subgraph associated with a node
        
        Parameters
        ----------
        graph_name : str
            Name of the graph.
        subgraph_name : str
            Name of the subgraph group.
        node_id : int
            The node id with which the subgraph is associated. For one subgraph group,
            a node can only be associated with a single subgraph.
        subgraph_nids : Tensor of integers
            Ids of the nodes in the subgraph. The tensor is of shape (N,), where N is
            the number of nodes in the subgraph.
        subgraph_eids : Tensor of integers
            Ids of the edges in the subgraph. The tensor is of shape (E,), where E is
            the number of edges in the subgraph.
        subgraph_nweights : Tensor of floats, optional
            Weights of the nodes in the subgraph, corresponding to subgraph_nids. The
            tensor can be reshaped as (N,), where N is the number of nodes in the
            subgraph. The weights should be in range [0, 1].
        subgraph_eweights : Tensor of floats, optional
            Weights of the edges in the subgraph, corresponding to subgraph_eids. The
            tensor can be reshaped as (E,), where E is the number of edges in the
            subgraph. The weights should be in range [0, 1].
        """
        assert graph_name in self.graph_names, \
            'Expect add_graph to be called first for graph {}'.format(graph_name)
        
        # Register the subgraph
        if subgraph_name not in self.graph_data[graph_name]['subgraph_list']:
            self.graph_data[graph_name]['subgraph_list'].append(subgraph_name)
            self.graph_data[graph_name]['subgraphs'][subgraph_name] = {
                "name": subgraph_name,
                "success": True,
                "node_subgraphs": dict()
            }
            num_nodes = self.graph_data[graph_name]['num_nodes']
            for i in range(num_nodes):
                self.graph_data[graph_name]['subgraphs'][subgraph_name]["node_subgraphs"][i] = {
                    "nodes": [],
                    "nweight": [],
                    "eids": [],
                    "eweight": []
                }
        
        # GNNLens assumes the node and edge IDs to be sorted
        subgraph_nids = F.asnumpy(subgraph_nids)
        nid_order = np.argsort(subgraph_nids)
        subgraph_nids = subgraph_nids[nid_order]
        
        subgraph_eids = F.asnumpy(subgraph_eids)
        eid_order = np.argsort(subgraph_eids)
        subgraph_eids = subgraph_eids[eid_order]
        
        # Handle nweights
        if subgraph_nweights is None:
            subgraph_nweights = np.ones(len(subgraph_nids))
        else:
            subgraph_nweights = F.asnumpy(subgraph_nweights)
            subgraph_nweights = np.reshape(subgraph_nweights, (len(subgraph_nids),))
            subgraph_nweights = subgraph_nweights[nid_order]

        # Handle eweights
        if subgraph_eweights is None:
            subgraph_eweights = np.ones(len(subgraph_eids))
        else:
            subgraph_eweights = F.asnumpy(subgraph_eweights)
            subgraph_eweights = np.reshape(subgraph_eweights, (len(subgraph_eids),))
            subgraph_eweights = subgraph_eweights[eid_order]
        
        self.graph_data[graph_name]['subgraphs'][subgraph_name]["node_subgraphs"][node_id] = {
            "nodes": subgraph_nids.tolist(),
            "nweight": subgraph_nweights.tolist(),
            "eids": subgraph_eids.tolist(),
            "eweight": subgraph_eweights.tolist()
        }

    def close(self):
        """Finish dumping data."""
        # Dump data list (meta info)
        with open(self.logdir + '/dataset_list.json', 'w') as f:
            datasets = []
            for i, name in enumerate(self.graph_names):
                datasets.append({"id": i + 1, "name": name})
            json.dump({"datasets": datasets, "success": True}, f)
        
        for name in self.graph_names:
            graph_logdir = self._get_graph_logdir(name)

            # Dump model meta info
            with open(graph_logdir + '/model_list.json', 'w') as f:
                models = []
                for i, model_name in enumerate(self.graph_data[name]['model_list']):
                    models.append({"id": i + 1, "name": model_name})
                json.dump({"models": models, "success": True}, f)
            
            subgs = []
            for i, subg_name in enumerate(self.graph_data[name]['subgraph_list']):
                subgs.append({"id": i + 1, "name": subg_name})
                # Dump subgraph info
                with open(graph_logdir + '/subgraph_{}.json'.format(i + 1), 'w') as f:
                    json.dump(self.graph_data[name]['subgraphs'][subg_name], f)
            
            # Dump subgraph meta info
            with open(graph_logdir + '/subgraph_list.json', 'w') as f:
                json.dump({"subgraphs": subgs, "success": True}, f)
