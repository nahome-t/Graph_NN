from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from models import GCN, GfNN
from data_handler import test_accuracy

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Only one graph in this particular dataset so we just need
# to load the first element

learning_rate = 0.01 # Figure out good value?

num_epochs = 50 # Number of epochs over which data will be trained on,
# should eventually be changed so that it is a variable number which stops
# once 100% accuracy reached in training data, or we reach some max limit

hidden_layer_size = 16 # Size of hidden convolution layers (all same size)
in_features = data.num_features
out_features = dataset.num_classes
depth = int(input('Enter the depth of neural network: '))

if input('GfNN or GCN: ') == 'GCN':
    model = GCN(in_features=in_features,
                out_features=out_features,
                depth=depth,
                hidden_layer_size=hidden_layer_size)
else:
    model = GfNN(in_features=in_features,
                 out_features=out_features,
                 k=depth,
                 hidden_layer_size=hidden_layer_size)

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
        test_accuracy(model, data, epoch+1)  # +1 because epoch should start
        # from 1
    # test_accuracy(model, data, num_epochs)


train(model=model, data=data)