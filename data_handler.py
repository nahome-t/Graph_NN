# Contains functions used to read and write data
import torch
import numpy as np
from os.path import exists
from torch_geometric.datasets import Planetoid

def generate_mask(data_y, mask, group_size=20, num_classes=7, name="Cora"):
    # Picks group_size*num_classes examples from the data within the mask
    # given such that such tha each class is appears group_size times (
    # assuming it can find that many)

    if exists(f'benchmark_mask:{name}.npy'):
        print('Loading, benchmark mask, already exists...')
        return torch.from_numpy(np.load(f'benchmark_mask:{name}.npy'))

    data_y = data_y.tolist()
    mask = mask.tolist()
    test_mask = [False] * len(data_y)
    freq = {}

    # Initialises dictionary which will count how many of each class there is
    # so that we stop once we reach group size
    for i in range(num_classes):
        freq[i] = 0

    # Goes through the y values for the data and adds it to benchmark_mask if
    # it is in the mask given and we the number of that class is under the
    # group size we want to find
    for i, x in enumerate(data_y):
        # Only includes data that is in training data
        if mask[i] and freq[x] < group_size:
            freq[x] += 1
            test_mask[i] = True

    np.save(file=f'benchmark_mask:{name}', arr=np.array(test_mask))
    return torch.tensor(test_mask)


def count_frequency(filename='tensor2'):
    # Gets  data from file, changes it into 2D numpy tensor and effectively
    # counts how often each row or 'function' occurs
    tensor = np.loadtxt(fname=filename, delimiter=',', )
    dt = np.dtype((np.void, tensor.dtype.itemsize * tensor.shape[1]))
    b = np.ascontiguousarray(tensor).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(tensor.dtype).reshape(-1, tensor.shape[1])
    print(list(cnt))


def is_consistent(model, data):
    # Returns true if the model perfectly predicts the training data for a
    # dataset which can be filtered from entire dataset using dataset.mask()[]
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

def write_to_file(arr, fpath):
    # Takes in a 1d np array and then adds it to a csv as a row
    # arr = np.reshape(arr, (-1, 1))
    # print(np.reshape(arr, (-1, 1)))
    arr = np.reshape(arr, (1, -1))
    print(arr)
    with open(fpath, 'ab') as file:
        np.savetxt(fname=file, X=arr, delimiter=",", fmt='%d')

def test_accuracy(model, data, epoch_num=None, on_training_data = True):
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
