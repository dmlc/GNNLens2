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

Next, we train two graph convolutional networks (GCN) for node classification, GCN_L1 (GCN with one layer) and GCN_L2 (GCN with two layers). Once trained, we retrieve the predicted node classes and dump them to local files

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv

# Define a class for GCN
class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 num_classes,
                 num_layers):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, num_classes))
        for _ in range(num_layers - 1):
            self.layers.append(GraphConv(num_classes, num_classes))

    def forward(self, g, h):
        for layer in self.layers:
            h = layer(g, h)
        return h

# Define a function to train a GCN with the specified number of layers 
# and return the predictions
def train_gcn(g, num_layers, num_classes):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    model = GCN(in_feats=features.shape[1],
                num_classes=num_classes,
                num_layers=num_layers)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
  
    num_epochs = 200
    model.train()
    for _ in range(num_epochs):
        logits = model(g, features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      
    model.eval()
    predictions = model(g, features)
    _, predicted_classes = torch.max(predictions, dim=1)
    return predicted_classes

print("Training GCN with one layer...")
predictions_one_layer = train_gcn(graph, num_layers=1, num_classes=num_classes)
print("Training GCN with two layers...")
predictions_two_layers = train_gcn(graph, num_layers=2, num_classes=num_classes)
# Dump the predictions to local files
writer.add_model(graph_name='Cora', model_name='GCN_L1',
                 nlabels=predictions_one_layer)
writer.add_model(graph_name='Cora', model_name='GCN_L2',
                 nlabels=predictions_two_layers)
# Finish dumping
writer.close()
```
