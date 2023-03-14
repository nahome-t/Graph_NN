import torch.nn as nn
from torch import matmul, zeros
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import add_self_loops, degree
import torch.functional as F


class GCN(nn.Module):

    # Creates a GCN of arbitrary depth, note the first layer goes from
    # in_features->hidden_layer_size, each subsequent layer from there on
    # keeps the same hidden layer size until the final layer which maps to
    # the number of features you want to update (7 in the case of Cora)

    def __init__(self, in_features, out_features, hidden_layer_size, depth=2):
        super().__init__()
        self.in_conv = GCNConv(in_features, hidden_layer_size)
        self.convs = nn.ModuleList(
            [GCNConv(hidden_layer_size, hidden_layer_size) for _
             in range(depth - 2)])
        self.out_conv = GCNConv(hidden_layer_size, out_features)

    def forward(self, data):
        h, edge_index = data.x, data.edge_index
        h = self.in_conv(h, edge_index)
        h = F.relu(h)
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.Relu(h)
        h = self.conv2(h, edge_index)
        return F.log_softmax(h, dim=1)


class Norm_adj(MessagePassing):
    # Effectively multiplying the feature vectors by the normalized
    # augmented adjacency matrix or effectively a graph convolutional layer
    # with W set to
    # identity and no bias

    # Equivalent to adding self loop to each node, summing over neighbours
    # feature vectors such that two nodes of degree n1, n2 will also be
    # normalised by (n1*n2)^(-0.5)

    def __init__(self):
        super().__init__(aggr='add') # Aggregates via adding

    def forward(self, x, edge_index):
        # Adds self loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        row, col = edge_index

        deg = degree(col, x.size(0), dtype=x.dtype)  # Calculates degree for
        # each node
        deg_inv_sqrt = deg.pow(-0.5)

        # If node has 0 degree then normalisations set to 0 (shouldn't be
        # case since self loops added anyway)? but implemented in graph
        # convolution
        deg_inv_sqrt[deg_inv_sqrt==float('inf')] = 0

        norm = deg_inv_sqrt[row]*deg_inv_sqrt[col]
        # Propagates values with normalisation
        out = self.propagate(edge_index=edge_index, x=x, norm=norm)
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j # Normalizes value by normalisation
        # constant calculated in forward layer, note the view method insures
        # the tensor has the correct shap


class GfNN(nn.Module):
    # Implementation of gfNN as shown in https://arxiv.org/pdf/1905.09550.pdf
    # with difference being log_softmax being applied instead of softmax.
    # Note also we have that the augmented normalised adjacency matrix is
    # applied k times
    def __init__(self, in_features, hidden_layer_size, out_features, k):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden_layer_size)
        self.classifier = nn.Linear(in_features=hidden_layer_size,
                                out_features=out_features, bias=False)
        self.adj = Norm_adj()
        #
        self.k = k

    def forward(self, data):
        h, edge_index = data.x, data.edge_index

        # Note the adjacency layer applied k-1 times as the following
        # convolutional layer is equivalent to an adjacency layer followed by a
        # linear layer
        for _ in range(self.k-1):
            h = self.adj(h, edge_index)  # Can be applied multiple times
        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.classifier(h)
        return F.log_softmax(h, dim=1)

