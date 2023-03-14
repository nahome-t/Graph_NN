from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
from models import GCN, GfNN

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Only one graph in this particular dataset so we just need
# to load the first element


learning_rate = 0.01 # Figure out good value?
num_epochs = 50
hidden_layer_size = 16
in_features = data.num_features
out_features = dataset.num_classes
depth = 2

if input('GfNN or GCN') == 'GCN':
    model = GCN(in_features=in_features, out_features=out_features, depth=depth)
else:
    model = GfNN(in_features=in_features, out_features=out_features,
                 k=depth)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
data = data.to(device)

