from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
import numpy as np
from models import GCN, GfNN
import time
from data_handler import is_consistent, generate_mask


learning_rate = 0.01 # Figure out good value?
max_epochs = 50 # Model trained to 100% accuracy on training dataset,
# maximum epochs represents
hidden_layer_size = 16
GNN_type = 'GCN'
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0] # Only one graph in this particular dataset so we just need
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
                    hidden_layer_size=hidden_layer_size)
    return model.to(device)



def train2(model, data, mask):
    # A slightly altered training method, model is trained to 100% accuracy
    # on training data, this model is then applied to the mask given by one
    # of the parameters (in this case the generated test mask), the output of
    # this is then outputted

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for i in range(max_epochs):
        if is_consistent(model=model, data=data):
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

    print(f'Model not found... trying again')
    # Tries again to find it if function not found within 50 epochs
    return train2(model, data, mask)


def run_simulation(dataset_name, train_it, test_num, model_type, model_depth):
    # runs model multiple

    # dataset_name: in case of cora just 'cora'
    # train_it: True/False whether we want to train neural network before
    # applying it to dataset
    # test size: the amount of times we want to apply our model to generated
    # neural network
    # model_type: 'GCN' or 'GfNN'
    # model_depth: depth of neural network
    # Potentially use format cora_trained_GFNN_2 or just use a header!!

    start_time = time.time()
    # Generates mask, or loads it if int can find
    test_mask = generate_mask(data_y=data.y, mask=data.test_mask, name=dataset_name)

    # Stores the result of test
    aggr = np.zeros((test_num, test_mask.sum()))
    for k in range(test_num):
        model = generate_model(GNN_type='GCN', depth=2)
        arr = train2(model, data, test_mask).detach().cpu().numpy()
        arr = np.transpose(arr)
        aggr[k, :] = arr
        print(f'Done {k + 1}/{test_num}')
    np.savetxt('tensor', np.array(aggr), delimiter=',', fmt='%d')

    print(f'It took {time.time()-start_time} to finish this program and '
          f'generate {test_num} Neural networks')

run_simulation()

