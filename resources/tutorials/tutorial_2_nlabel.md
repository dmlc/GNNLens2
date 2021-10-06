# Tutorial 2: Ground truth and predicted node labels

The nodes in a graph can be associated with a label like node type or node class. For the task of multiclass node classification, you can have ground truth node labels and node labels predicted from different models. GNNLens2 allows coloring nodes based on node labels in graph visualization and comparing node labels from different sources.

## Data preparation

First, we load DGL’s built-in Cora dataset and retrieve its graph structure, node labels (classes) and number of node classes.

```python
from dgl.data import CoraGraphDataset

dataset = CoraGraphDataset()
graph = dataset[0]
nlabels = graph.ndata['label']
num_classes = dataset.num_classes
```

We dump them to a local file that GNNLens2 can read. Compared with [the previous section](./tutorial_1_graph.md), we additionally dump the node classes and the number of node classes. 

```python
from gnnlens import Writer

# Specify the path to create a new directory for dumping data files.
writer = Writer('tutorial_nlabel')
writer.add_graph(name='Cora', graph=graph, 
                 nlabels=nlabels, num_nlabel_types=num_classes)
```
