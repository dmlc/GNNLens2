import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.nn.pytorch import GraphConv
from viswriter import VisWriter

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

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
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
    
    acc = evaluate(model, g, features, labels, g.ndata['test_mask'])
    print("Test accuracy {:.2%}".format(acc))
    
    model.eval()
    predictions = model(g, features)
    _, predicted_classes = torch.max(predictions, dim=1)
    return predicted_classes

def main(args):
    citation_networks = {
        'Cora': CoraGraphDataset,
        'Citeseer': CiteseerGraphDataset,
        'Pubmed': PubmedGraphDataset
    }
    dataset = citation_networks[args.data]()
    graph = dataset[0]
    nlabels = graph.ndata['label']
    num_classes = dataset.num_classes
    
    writer = VisWriter('tutorial_nlabel')
    writer.add_graph(name='Cora', graph=graph, 
                     nlabels=nlabels, num_nlabel_types=num_classes)
    
    print("Training GCN with one layer...")
    predictions_one_layer = train_gcn(graph, num_layers=1, num_classes=num_classes)
    writer.add_model(graph_name='Cora', model_name='GCN_L1', nlabels=predictions_one_layer)
    print("Training GCN with two layers...")
    predictions_two_layers = train_gcn(graph, num_layers=2, num_classes=num_classes)
    writer.add_model(graph_name='Cora', model_name='GCN_L2', nlabels=predictions_two_layers)

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data', '-d', choices=['Cora', 'Citeseer', 'Pubmed'])
    args = parser.parse_args()

    main(args)
