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