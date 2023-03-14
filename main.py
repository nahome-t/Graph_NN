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

num_epochs = 50 # Number of epochs over which data will be trained on,
# should eventually be changed so that it is a variable number which stops
# once 100% accuracy reached in training data

hidden_layer_size = 16 # Size of hidden convolution layers (all same size)
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


def train(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    test_accuracy(model=model, data=data, epoch_num=0)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data)

        # data.train_mask/test_mask is a tensor of True/False values which
        # filter the dataset to only include training or testing data
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward() # Calculates the gradients
        optimizer.step() # Updates the model via gradient descent
        # test_accuracy(model, data, epoch+1) #+1 because epoch should start
        # from 1
    test_accuracy(model, data, num_epochs)

def test_accuracy(model, data, epoch_num=None, on_training_data = False):
    # Will test on the remaining data set
    if epoch_num == 0:
        print('CHECKING ACCURACY ON TRAINING DATA') if on_training_data else \
            print('CHECKING ACCURACY ON TESTING DATA')
    mask = data.train_mask if on_training_data else data.test_mask  # Masks
    # are pytorch tensors which are of the form [True, True, False,
    # ....] which such that the train_mask and test_mask filters the data by
    # training and testing data

    model.eval()
    with torch.no_grad():
        test_num = mask.sum()
        prediction = model(data).argmax(dim=1)
        correct = (prediction[mask] == data.y[mask]).sum()
        acc = int(correct)/test_num
        print(f'Epoch num: {epoch_num}, Accuracy {acc:.2f}, i.e. {correct}'
              f'/{test_num}')
    model.train()


train(model=model, data=data)