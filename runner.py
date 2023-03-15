from torch_geometric.datasets import Planetoid
import torch

import torch.nn.functional as F

from models import GCN, GfNN


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



def generate_mask(data, mask):
    # Picks 140 examples from the testing data such that each class is
    # picked 20 times, note the testing data is only a subset of the entire
    # data and so the mask for the testing data needs to be given

    # NOTE CLASS SIZE OF 20 SPECIFICALLY GEARED TOWARDS CORA DATASET

    CLASS_SIZE = 20

    data = data.tolist()
    mask = mask.tolist()
    test_mask = [False]*len(data)
    freq = {}
    for i in range(7):
        freq[i] = 0

    for i, x in enumerate(data):
        if mask[i] and freq[x]<CLASS_SIZE:
            freq[x]+=1
            test_mask[i] = True


    # indicies = [i for i in range(len(data)) if test_mask[i] == True]
    # print('Test indicies')

    return torch.tensor(test_mask)


def is_consistent(model, data):
    # Returns true if the model perfectly predicts the data, otherwise
    # returns false
    mask = data.train_mask
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
    # A slightly altered training method, model is trained to 100% accuracy
    # on training data, this model is then applied to the mask given by one
    # of the parameters (in this case the generated test mask, the output of
    # this is then outputted

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
    # Tries again to find it if function not found within 50 epochs
    return train2(model, data, mask)


def run_simulation(filename):
    return None

    # test_mask = generate_mask(data.y, data.test_mask)
    #
    # test_size = 40  # How many times do we want to run each test
    # for k in range(test_size):
    #     model = model.to(device)
    #     train2(model, data, test_mask)


