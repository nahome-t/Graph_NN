from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from os.path import exists
from models import GCN, GfNN, NormAdj
import time
from data_handler import is_consistent, generate_mask, get_file_name, \
    write_to_file, applyAdjLayer
from pyinstrument import Profiler
import numpy as np

learning_rate = 0.01  # Figure out good value?
max_epochs = 200  # Model trained to 100% accuracy on training dataset,
# maximum epochs represents
hidden_layer_size = 16
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]  # Only one graph in this particular dataset so we just need
# to load the first element

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)


def generate_model(GNN_type, depth):
    # Generates a model using the dataset in the function and the given
    # parameters
    if GNN_type == 'GCN':
        model = GCN(in_features=data.num_features,
                    out_features=dataset.num_classes,
                    depth=depth,
                    hidden_layer_size=hidden_layer_size)
    elif GNN_type == 'GfNN':
        model = GfNN(in_features=data.num_features,
                     out_features=dataset.num_classes,
                     k=depth,
                     hidden_layer_size=hidden_layer_size,
                     adj_layer=False)
        # Since we want the depth to be the number of adjacency layers we
        # have we can effectively pre-compute the layers beforehand

    return model.to(device)


def train2_perf(model, data, mask):
    # A slightly altered training method, model is trained to 100% accuracy
    # on training data, this model is then applied to the data with the mask
    # given by one of the parameters (in this case the generated test mask),
    # the output of this is then outputted

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for i in range(max_epochs):
        if is_consistent(model=model, data=data):
            print(f'A model with 100% accuracy was found at epoch {i}')
            with torch.no_grad():
                return model(data)[mask].argmax(dim=1).detach().cpu().numpy()

        optimizer.zero_grad()
        output = model(data)

        # data.train_mask/test_mask is a tensor of True/False values which
        # filter the dataset to only include training or testing data
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()  # Calculates the gradients
        optimizer.step()  # Updates the model via gradient descent

    model.eval()

    print(f'Model not found within max epochs: {max_epochs} ... trying again')

    # Tries again to find it if function not found within max epochs, actually
    # this won't try and find one from scratch as the model has some
    # effectively learnt parameters that would need to be reset
    return np.array([])


def run_simulation(dataset_name, train_it, test_num, model_type, model_depth):
    # runs model multiple

    # dataset_name: in case of cora just 'Cora'
    # train_it: True/False whether we want to train neural network before
    # applying it to dataset
    # test size: the amount of times we want to apply our model to generated
    # neural network
    # model_type: 'GCN' or 'GfNN'
    # model_depth: depth of neural network
    # Potentially use format cora_trained_GFNN_2 or just use a header!!

    start_time = time.time()

    # If we have that we want to train GNN easier to pre-compute initial
    # adjacency layers and so
    data_used = applyAdjLayer(data, model_depth) \
        if model_type == "GfNN" else data

    # Generates mask, or loads it if it exists within file system,
    generated_mask = generate_mask(data_y=data.y, mask=data.train_mask,
                                   group_size=20, num_classes=7, name="Cora")

    fname = get_file_name(dataset_name, train_it, model_type,
                          model_depth)

    # Creates file if it doesn't exist
    if not exists(fname):
        open(fname, "x")
    k = 0

    while k < test_num:
        model = generate_model(GNN_type=model_type, depth=model_depth)
        # If we want our test to train neural network it'll train it,
        # otherwise it'll just apply our model to data
        if train_it:
            arr = train2_perf(model, data_used, generated_mask)
        else:
            arr = model(data_used)[generated_mask].argmax(dim=1)
        # print(arr)
        print(f'Done {k + 1}/{test_num}')

        if arr.size == 0:
            print('gg')
            continue

        write_to_file(arr, fname)
        k += 1

    print(
        f'It took {(time.time() - start_time):.3f} to finish this program and '
        f'generate {test_num} '
        f'{"trained" if train_it else "random"} neural networks')


# Used to check training times for each type of neural network


run_simulation(dataset_name="Cora", train_it=True, test_num=100,
               model_type='GfNN', model_depth=10)
