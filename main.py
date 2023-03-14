from torch_geometric.datasets import Planetoid
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import time
# from models import GCN, GfNN

learning_rate = 0.01 # Figure out good value?
num_epochs = 50
hidden_layer_size = 16

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Only one graph in this particular dataset so we just need
# to load the first element
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# model, data = GfNN().to(device), data.to(device)

