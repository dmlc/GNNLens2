"""Dump data for GNNVis"""
import dgl.backend as F
import json
import numpy as np
import os
from dgl import DGLError

class VisWriter():
    """
    Parameters
    ----------
    logdir: str
        Path to create a new directory for dumping data files, which can 
        be either a relative path or an absolute path.
    """
    def __init__(self, logdir):
        os.makedirs(logdir, exist_ok=False)
        self.logdir = logdir
        self.data_name_to_id = dict()
        self.data_name_to_num_nodes = dict()
        self.data_name_to_num_edges = dict()
        self.data_name_to_model_name_to_id = dict()
        self.next_graph_id = 1
    
    def add_graph(self, name, graph, nlabels=None, num_nlabel_types=None, eweights=None):
        """
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
        if name in self.data_name_to_id:
            raise DGLError('Graph name {} has already been used.'.format(name))
        else:
            graph_logdir = os.path.join(self.logdir, str(self.next_graph_id))
            os.makedirs(graph_logdir)
            self.data_name_to_id[name] = self.next_graph_id
            self.data_name_to_model_name_to_id[name] = dict()
            self.next_graph_id += 1
        
        srcs, dsts = graph.edges()
        srcs = F.asnumpy(srcs).tolist()
        dsts = F.asnumpy(dsts).tolist()
        num_nodes = graph.num_nodes()
        self.data_name_to_num_nodes[name] = num_nodes
        num_edges = graph.num_edges()
        self.data_name_to_num_edges[name] = num_edges

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
            num_edges = graph.num_edges()
            for ew_name, ew_values in eweights.items():
                ew_values = F.asnumpy(ew_values)
                try:
                    ew_values = np.reshape(ew_values, (num_edges,))
                except:
                    raise DGLError('Edge weights should be able to be reshaped as (num_edges,)')
                eweights[ew_name] = ew_values.tolist()

        """
        # layout
        nx_graph = graph.to_networkx()
        pos = nx.spring_layout(nx_graph)
        pos = {k: v.tolist() for k, v in pos.items()}
        """

        graph_obj = {
            "name": name,
            "srcs": srcs,
            "dsts": dsts,
            "num_nodes": num_nodes,
            "nlabels": nlabels,
            "num_nlabel_types": num_nlabel_types,
            "eweights": eweights
            # "layout": pos
        }
        data_obj = {
            "graph_obj": graph_obj,
            "success": True
        }

        # Dump data list (meta info)
        with open(self.logdir + '/dataset_list.json', 'w') as f:
            datasets = []
            for g_name, g_id in self.data_name_to_id.items():
                datasets.append({
                    "id": g_id,
                    "name": g_name
                })
            json.dump({"datasets": datasets, "success": True}, f)
        
        # Dump empty model meta info
        with open(graph_logdir + '/model_list.json', 'w') as f:
            json.dump({"models": [], "success": True}, f)

        # Dump empty subgraph meta info
        with open(graph_logdir + '/subgraph_list.json', 'w') as f:
            json.dump({"subgraphs": [], "success": True}, f)

        # Dump graph data file
        with open(graph_logdir + '/graph.json', 'w') as f:
            json.dump(data_obj, f)

    def add_model(self, graph_name, model_name, nlabels, eweights=None):
        # TBD (doc)
        assert graph_name in self.data_name_to_id, \
            'Expect add_graph to be called first for graph {}'.format(graph_name)
        graph_logdir = os.path.join(self.logdir, str(self.data_name_to_id[graph_name]))
        
        nlabels = F.asnumpy(nlabels)
        num_nodes = self.data_name_to_num_nodes[graph_name]
        try:
            nlabels = np.reshape(nlabels, (num_nodes,))
        except:
            raise DGLError('Node labels should be able to be reshaped as (num_nodes,)')
        nlabels = nlabels.tolist()
        
        if eweights is None:
            eweights = dict()
        else:
            num_edges = self.data_name_to_num_edges[graph_name]
            for ew_name, ew_values in eweights.items():
                ew_values = F.asnumpy(ew_values)
                try:
                    ew_values = np.reshape(ew_values, (num_edges,))
                except:
                    raise DGLError('Edge weights should be able to be reshaped as (num_edges,)')
                eweights[ew_name] = ew_values.tolist()

        # Register the model
        num_models = len(self.data_name_to_model_name_to_id[graph_name]) + 1
        self.data_name_to_model_name_to_id[graph_name][model_name] = num_models
        with open(graph_logdir + '/model_list.json', 'w') as f:
            models = []
            for m_name, m_id in self.data_name_to_model_name_to_id[graph_name].items():
                models.append({
                    "id": m_id,
                    "name": m_name
                })
            json.dump({"models": models, "success": True}, f)
        
        # Dump model data file
        with open(graph_logdir + '/model_{}.json'.format(num_models), 'w') as f:
            model_obj = {
                "name": model_name,
                "nlabels": nlabels,
                "eweights": eweights
            }
            json.dump({"model_obj": model_obj, "success": True}, f)
