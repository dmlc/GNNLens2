from dgl.data import CoraGraphDataset, CiteseerGraphDataset
from viswriter import VisWriter

def main():
    writer = VisWriter('tutorial_graph')

    cora_dataset = CoraGraphDataset()
    cora_graph = cora_dataset[0]
    writer.add_graph(name='Cora', graph=cora_graph)

    citeseer_dataset = CiteseerGraphDataset()
    citeseer_graph = citeseer_dataset[0]
    writer.add_graph(name='Citeseer', graph=citeseer_graph)

if __name__ == '__main__':
    main()
