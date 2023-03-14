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




