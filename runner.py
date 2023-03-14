from torch_geometric.datasets import Planetoid
import torch

import torch.nn as nn
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F
import time
from models import GCN, GfNN


learning_rate = 0.01 # Figure out good value?
max_epochs = 50
hidden_layer_size = 16
depth = 2

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Only one graph in this particular dataset so we just need
# to load the first element

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = GCN(in_features=data.num_features,
            out_features=dataset.num_classes,
            depth=depth,
            hidden_layer_size=hidden_layer_size)
model, data = model.to(device), data.to(device)

# class DataHandler()

def generate_mask(data, mask):
    # Picks 140 examples from the testing data such that each class is
    # picked 20 times, note the testing data is only a subset of the entire
    # data and so the mask for the testing data needs to be given

    data = data.tolist()
    mask = mask.tolist()
    test_mask = [False]*len(data)
    freq = {}
    for i in range(7):
        freq[i] = 0

    for i, x in enumerate(data):
        if mask[i] and freq[x]<20:
            freq[x]+=1
            test_mask[i] = True


    # indicies = [i for i in range(len(data)) if test_mask[i] == True]
    # print('Test indicies')

    return torch.tensor(test_mask)


def is_consistent(model, data, mask):
    # Returns true if the model perfectly predicts the data, otherwise
    # returns false

    model.eval()
    with torch.no_grad():
        test_num = mask.sum()
        prediction = model(data).argmax(dim=1)
        correct = (prediction[mask] == data.y[mask]).sum()
        if int(correct) == mask.sum():
            return True
        else:
            return False

    model.train()  # Sets it back to training mode by default


def train2(model, data, mask):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for i in range(max_epochs):
        if is_consistent(model=model, data=data, mask=data.train_mask) == True:
            print(f'A model with 100% accuracy was found at epoch {i}')
            with torch.no_grad():
                return model(data)[mask].argmax(dim=1)

        optimizer.zero_grad()
        output = model(data)

        # data.train_mask/test_mask is a tensor of True/False values which
        # filter the dataset to only include training or testing data
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Calculates the gradients
        optimizer.step()  # Updates the model via gradient descent

    model.eval()

    print(f'Model not found...')

test_mask = generate_mask(data.y, data.test_mask)

num_tests = 1
test_size = 10  # How many tensors each test will store

test_size = 50
for k in range(test_size):
    model = GCN(in_features=data.num_features,
                out_features=dataset.num_classes,
                depth=depth,
                hidden_layer_size=hidden_layer_size)
    model = model.to(device)
    print(train2(model, data, test_mask))


