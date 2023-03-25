# Contains functions used to read and write data
from pathlib import Path
import torch
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from models import NormAdj

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

def get_file_name(dataset_name, train_it, model_type, model_depth):
    # Gets the file name that an output should be saved to given the name of
    # a dataset, whether its trained or not and the model type or depth
    extension = f'/output/{dataset_name}' \
               f'_{"trained" if train_it else "random"}_' \
            f'{model_type}_{model_depth}'
    program_path = Path(__file__)
    fname = str(program_path.parent.absolute()) + extension
    return fname


def count_frequency(filename='tensor2', width=None, binarised=False):
    # Gets  data from file, changes it into 2D numpy tensor and effectively
    # counts how often each row or 'function' occurs
    tensor = np.loadtxt(fname=filename, delimiter=',')
    if binarised:
        tensor = tensor%2
    print(tensor.shape)

    if width:
        tensor = tensor[:, :width]

    print(tensor.shape)

    unq, cnt = np.unique(tensor, return_counts=True, axis=0)
    print(unq)
    return -np.sort(-cnt)




def is_consistent(model, data):
    # Returns true if the model perfectly predicts the training data for a
    # dataset which can be filtered from entire dataset using dataset.mask()[]
    mask = data.train_mask
    model.eval()
    with torch.no_grad():
        prediction = model(data).argmax(dim=1)
        correct = (prediction[mask] == data.y[mask]).sum()
        if int(correct) == mask.sum():
            return True
        else:
            return False

    model.train()  # Sets it back to training mode by default

def write_to_file(arr, fpath):
    # Takes in a 1d np array and then adds it to a csv as a row
    arr = np.reshape(arr, (1, -1))
    # print(arr)
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


def applyAdjLayer(data, depth):
    # Applies normalised adjacency layer depth amount of times, maybe should
    # be in models rather than data_handler
    adj_layer = NormAdj()
    smoothed_data = data.clone()
    # Effectively applies adj layer depth amount of times
    for _ in range(depth):
        smoothed_data.x = adj_layer(smoothed_data.x, data.edge_index)
    return smoothed_data


def produce_rankVProb_plot(*arrays, labels = None, title="Rank vs "
                                                         "Probability",
                           xlabel="Rank", ylabel="Probability", log_scale=True):
    max_length = 0
    arrays = list(arrays)
    for i in range(len(arrays)):
        max_length = max(max_length, arrays[i].size)
        arrays[i] = 1/np.sum(arrays[i])*arrays[i] # Normalises array

    print(arrays)

    rank = np.arange(max_length)
    for i in range(len(arrays)):
        # Makes sure all the arrays are the same length
        l = max_length-arrays[i].size
        arrays[i] = np.concatenate((arrays[i], np.zeros(l)))


        if labels:
            plt.plot(rank, arrays[i], label=labels[i])
            plt.legend()
        else:
            plt.plot(rank, arrays[i])

        plt.xlabel('Rank')
        plt.ylabel('Probability')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.show()


WIDTH = 15

fname1 = get_file_name("Cora", False, "GCN", 2)
freq1 = count_frequency(fname1, width=WIDTH)
freq2 = count_frequency(fname1, WIDTH*2)
freq3 = count_frequency(fname1, WIDTH*4)
produce_rankVProb_plot(freq1, freq2, freq3, labels=["Width: 35", "Width: 70",
                                              "Width: 140"])

# produce_rankVProb_plot(freq1)
# fname2 = get_file_name("Cora", True, "GfNN", 6)
# freq2 = count_frequency(fname2, width=WIDTH)

# produce_rankVProb_plot(freq1, freq2, labels=["GCN, depth 10", "GfNN, depth 10"])

# print(get_file_name('Cora', False, 'GFN', 10))



# print(list(sm))
# print("-------------------------")
# print("data after")
# print(list(nsm))
# print(data.x[data.train_mask][0])
