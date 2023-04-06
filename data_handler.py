# Contains functions used to read and write data
from pathlib import Path
import torch
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from models import NormAdj
from torch_geometric.datasets import Planetoid

def generate_mask(data_y, mask, group_size=20, num_classes=7, name="Cora"):
    # Picks group_size*num_classes examples from the data within the mask
    # given such that such tha each class is appears group_size times (
    # assuming it can find that many)

    if exists(f'benchmark_mask:{name}_{group_size}_{num_classes}.npy'):
        print('Loading, benchmark mask, already exists...')
        return torch.from_numpy(np.load(f'benchmark_mask:{name}_'
                                        f'{group_size}_{num_classes}.npy'))

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

    np.save(file=f'benchmark_mask:{name}_{group_size}_{num_classes}.npy',
            arr=np.array(test_mask))
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


def count_frequency(filename='tensor2', binarised=False,
                    perm_inv=False, mask=None):
    # Gets  data from file, changes it into 2D numpy tensor and effectively
    # counts how often each row or 'function' occurs
    tensor = np.loadtxt(fname=filename, delimiter=',')
    if binarised:
        tensor = tensor%2

    if perm_inv:
        tensor = permutational_order(tensor)
    print(tensor.shape)

    if mask is not None:
        tensor = tensor[:, mask] # Applies the mask to each row

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


def calc_row_err(probability, test_size, depth=10, z=1):
    # Given a set of probabilities and the number of tests needed to find
    # those probabilities returns the error on the data via the Wilson score
    # interval
    err = np.zeros((2, probability.size))
    err[0] = probability
    err[1] = probability
    for i in range(depth):
        err[0] = probability - z*np.sqrt(err[0] * (1 - err[0]) /
                                    test_size)
        err[1] = probability + z*np.sqrt(err[1] * (1 - err[1]) /
                                    test_size)

    return err

def produce_rankVProb_plot(*arrays, labels = None,
                           title="Rank vs Probability",
                           xlabel="Rank",
                           ylabel="Probability",
                           log_scale=True,
                           cumulative=False,
                           error = False):
    max_length = 0
    arrays = list(arrays)

    test_size = np.zeros(len(arrays))
    for i in range(len(arrays)):
        max_length = max(max_length, arrays[i].size)
        test_size[i] = np.sum(arrays[i])
        arrays[i] = 1/np.sum(arrays[i])*arrays[i] # Normalises array
        print(np.sum(arrays[i]))

    rank = np.arange(1, max_length + 1)  # Rank starts with 1
    err_val = np.zeros((len(arrays), max_length))

    print("THIS")



    for i in range(len(arrays)):
        # Makes sure all the arrays are the same length
        l = max_length-arrays[i].size
        arrays[i] = np.concatenate((arrays[i], np.zeros(l)))

        if error:
            e2 = calc_row_err(arrays[i], test_size[i], depth=5)
            plt.fill_between(rank, e2[0], e2[1], alpha=0.3)
        if cumulative:
            arrays[i] = np.cumsum(arrays[i])

        if labels:
            plt.plot(rank, arrays[i], label=labels[i])
            plt.legend()
        else:
            plt.plot(rank, arrays[i])
            print('hehe')

        plt.xlabel('Rank')
        plt.ylabel('Probability')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.show()


def permutational_order(arr):
    for i in range(arr.shape[0]):
        # print("ARRAY")
        # print(arr[i])
        unique_vals = np.unique(arr[i])
        unq, idx = np.unique(arr[i], return_index=True)
        val_map = {val: j for j, val in enumerate(arr[i, np.sort(idx)])}

        arr[i] = np.array([val_map[val] for val in arr[i]])
    return arr

def red_mask_for_cora(group_size):
    dataset = Planetoid(root='/tmp/Cora', name='Cora')
    data = dataset[0]

    # # This is the mask with 20 lots of each class
    m1 = generate_mask(data_y=data.y, group_size=20, num_classes=7,
                       mask=data.train_mask).numpy()

    # This is subset of original mask
    m2 = generate_mask(data_y=data.y, group_size=group_size, num_classes=7,
                       mask=m1).numpy()
    print(np.unique(data.y[m2].numpy(), return_counts=True))
    return m2[m1]



fname1 = get_file_name("Cora", False, "GfNN", 2)
fname2 = get_file_name("Cora", False, "GCN", 2)
freq1 = count_frequency(fname1, perm_inv=True, mask=red_mask_for_cora(3))
freq2 = count_frequency(fname1, mask=red_mask_for_cora(3))

# produce_rankVProb_plot(freq1)
# freq2 = count_frequency(fname1, perm_inv=False)
# freq3 = count_frequency(fname1, WIDTH*4, binarised=True)
# produce_rankVProb_plot(freq1, freq2, freq3, labels=["Width: 35", "Width: 70",
#                                               "Width: 140"], log_scale=True,
#                        cumulative=True)

produce_rankVProb_plot(freq1, freq2, labels=["f1", "f2"],
                       error=False)
# fname2 = get_file_name("Cora", True, "GfNN", 6)
# freq2 = count_frequency(fname2, width=WIDTH)

# produce_rankVProb_plot(freq1, freq2, labels=["GCN, depth 10", "GfNN, depth 10"])
# produce_rankVProb_plot(freq2, error=True)
# print(get_file_name('Cora', False, 'GFN', 10))



# print(list(sm))
# print("-------------------------")
# print("data after")
# print(list(nsm))
# print(data.x[data.train_mask][0])
