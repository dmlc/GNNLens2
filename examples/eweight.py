import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from viswriter import VisWriter

import dgl.function as fn
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        # input projection
        self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0]))
        # hidden layers
        for l in range(1, num_layers - 1):
            # due to multi-head, in_dim = num_hidden * number of heads in the previous layer
            self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l]))
        # output projection
        self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1]))

    def forward(self, g, h):
        attns = []
        for l in range(self.num_layers - 1):
            h, attn = self.gat_layers[l](g, h, get_attention=True)
            h = h.flatten(1)
            attns.append(attn)
        # output projection
        logits, attn = self.gat_layers[-1](g, h, get_attention=True)
        logits = logits.mean(1)
        attns.append(attn)
        return logits, attns

def evaluate(model, g, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)
    
def convert_attns_to_dict(attns):
    attn_dict = {}
    for layer, attn_list in enumerate(attns):
        attn_list = attn_list.squeeze(2).transpose(0, 1)
        for head, attn in enumerate(attn_list):
            head_name = "L{}_H{}".format(layer, head)
            attn_dict[head_name] = attn
    return attn_dict    

def train_gat(g, num_layers, heads, num_classes):
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    model = GAT(num_layers=num_layers,
                in_dim=features.shape[1],
                num_hidden=8,
                num_classes=num_classes,
                heads=heads)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    
    num_epochs = 35
    model.train()
    for epochs in range(num_epochs):
        logits, _ = model(g, features)
        loss = loss_func(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc = evaluate(model, g, features, labels, g.ndata['test_mask'])
    print("Test accuracy {:.2%}".format(acc))
    
    model.eval()
    predictions, attns = model(g, features)
    _, predicted_classes = torch.max(predictions, dim=1)
    attn_dict = convert_attns_to_dict(attns)

    return predicted_classes, attn_dict


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
    
    # Generate two types of random edge weights
    confidence = torch.rand(graph.num_edges(),)
    strength = torch.rand(graph.num_edges(),)
    
    # Specify the path to create a new directory for dumping data files.
    writer = VisWriter('tutorial_eweight')
    
    writer.add_graph(name='Cora', graph=graph, nlabels=nlabels, num_nlabel_types=num_classes, 
                     eweights={'confidence': confidence, 'strength': strength})
    print("Training GAT with two layers...")
    predictions_gat_two_layers, attn_dict_two_layers = train_gat(
        graph, num_layers=2, heads=[2,1], num_classes=num_classes)
    writer.add_model(graph_name='Cora', model_name='GAT_L2', 
                     nlabels=predictions_gat_two_layers, eweights=attn_dict_two_layers)

    print("Training GAT with three layers...")
    predictions_gat_three_layers, attn_dict_three_layers = train_gat(
        graph, num_layers=3, heads=[4,2,1], num_classes=num_classes)
    writer.add_model(graph_name='Cora', model_name='GAT_L3', 
                     nlabels=predictions_gat_three_layers, eweights=attn_dict_three_layers)


if __name__ == '__main__':
    from argparse import ArgumentParser
    import shutil
    parser = ArgumentParser()
    parser.add_argument('--data', '-d', choices=['Cora', 'Citeseer', 'Pubmed'], default="Cora")
    args = parser.parse_args()
    shutil.rmtree('tutorial_gat_attention', ignore_errors=True)
    main(args)
