import dgl
import torch

from dgl.data import CoraGraphDataset
from viswriter import VisWriter

def extract_subgraph(g, seed_node, num_hops=3):
    seed_nodes = [seed_node]
    for _ in range(num_hops - 1):
        sg = dgl.in_subgraph(g, seed_nodes)
        src, dst = sg.edges()
        seed_nodes = torch.cat([src, dst]).unique()
    sg = dgl.in_subgraph(g, seed_nodes, relabel_nodes=True)
    return sg

def main():
    writer = VisWriter('tutorial_subgraph')
    
    dataset = CoraGraphDataset()
    graph = dataset[0]
    nlabels = graph.ndata['label']
    num_classes = dataset.num_classes
    writer.add_graph(name='Cora', graph=graph, 
                     nlabels=nlabels, num_nlabel_types=num_classes)
    
    first_subgraph = extract_subgraph(graph, seed_node=0)
    second_subgraph = extract_subgraph(graph, seed_node=10)
    writer.add_subgraph(graph_name='Cora', subgraph_name='syn', node_id=0, 
                        subgraph_nids=first_subgraph.ndata[dgl.NID],
                        subgraph_eids=first_subgraph.edata[dgl.EID],
                        subgraph_nweights=torch.randn(first_subgraph.num_nodes()),
                        subgraph_eweights=torch.randn(first_subgraph.num_edges()))
    writer.add_subgraph(graph_name='Cora', subgraph_name='syn', node_id=10, 
                        subgraph_nids=second_subgraph.ndata[dgl.NID],
                        subgraph_eids=second_subgraph.edata[dgl.EID],
                        subgraph_nweights=torch.randn(second_subgraph.num_nodes()),
                        subgraph_eweights=torch.randn(second_subgraph.num_edges()))
    writer.close()

if __name__ == '__main__':
    main()
