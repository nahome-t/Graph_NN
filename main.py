import numpy as np
from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from models import GCN, GfNN
from data_handler import test_accuracy, get_file_name, write_to_file
from synthetic_data import make_data
from data_handler import generate_mask
# dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
# data = dataset[0] # Only one graph in this particular dataset so we just need
# to load the first element


data = torch.load(f='synth3')


# data = make_data(n_points=1000, train_size=120, test_size=120)

# data = torch.load(f='/Users/nahometewolde/PycharmProjects/Graph_NN'
#                     '/synthetic_torch_trained')


print(f' training: {torch.sum(data.train_mask)}, of which '
      f'{data.y[data.train_mask].sum()} belong to one of the classse')
print(f' testing: {torch.sum(data.test_mask)} of which '
      f'{data.y[data.test_mask].sum()} belongs to one of the classes')

print(data)
learning_rate = 0.005 # Figure out good value?

num_epochs = 300 # Number of epochs over which data will be trained on,
# should eventually be changed so that it is a variable number which stops
# once 100% accuracy reached in training data, or we reach some max limit

hidden_layer_size = 128  # Size of hidden convolution layers (all same size)
in_features = 2
out_features = 2
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

def train(model, data, v2=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    test_accuracy(model=model, data=data, epoch_num=0, on_training_data=True)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        output = model(data)

        # data.train_mask/test_mask is a tensor of True/False values which
        # filter the dataset to only include training or testing data
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Calculates the gradients
        optimizer.step()  # Updates the model via gradient descent
        test_accuracy(model, data, epoch+1, on_training_data=True)  # +1
        # because epoch should start
        # from 1
    # test_accuracy(model, data, num_epochs)

train(model=model, data=data)

torch.save(data, f='Synth2')
# def runner2(n_points, pos_scale, class_var_scale, train_mask_size,
#             near_neighbours, model_type, model_depth, test_num):
#
#
#     data = generate_data(n_points=n_points, pos_scale=pos_scale,
#                          class_var_scale=class_var_scale,
#                          train_mask_size=train_mask_size,
#                          near_neighbours=near_neighbours)
#     fname = get_file_name('Synth', train_it=True, model_type=model_type,
#                           model_depth=model_depth)
#
#
#     for i in range(test_num):
#         accuracy = np.zeros(num_epochs+1) # Will include 0 as epoch
#         if model_type == 'GCN':
#             model = GCN(in_features=in_features,
#                         out_features=out_features,
#                         depth=depth,
#                         hidden_layer_size=hidden_layer_size)
#         else:
#             model = GfNN(in_features=in_features,
#                          out_features=out_features,
#                          k=depth,
#                          hidden_layer_size=hidden_layer_size)
#
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#         model.train()
#         accuracy[0] = test_accuracy(model=model, data=data, epoch_num=0,
#                        on_training_data=False)
#
#         for epoch in range(num_epochs):
#             optimizer.zero_grad()
#             output = model(data)
#
#             # data.train_mask/test_mask is a tensor of True/False values which
#             # filter the dataset to only include training or testing data
#             loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
#             loss.backward()  # Calculates the gradients
#             optimizer.step()  # Updates the model via gradient descent
#             accuracy[epoch+1] = test_accuracy(model, data, epoch+1,
#                                   on_training_data=False)  # +1
#             print(accuracy)
#             write_to_file(accuracy, fpath=fname)
#
# runner2(n_points=5000, pos_scale=10, class_var_scale=10,
#         train_mask_size=3000, near_neighbours=4, model_type='GCN',
#         model_depth=4, test_num=4)
