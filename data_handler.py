# Contains functions used to read and write data
import os
from pathlib import Path
import pandas as pd
import torch, math
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from models import NormAdj
from torch_geometric.datasets import Planetoid
from os import listdir


def generate_mask(data_y, mask, num_classes, name, group_size=20):
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


def get_file_name(dataset_name, train_it, model_type, model_depth, rank=None,
                  prefix=None):
    # Gets the file name that an output should be saved to given the name of
    # a dataset, whether its trained or not and the model type or depth

    if prefix == None:
        prefix = '/output/'

    extension = f'{prefix}{dataset_name}' \
                f'_{"trained" if train_it else "random"}_' \
                f'{model_type}_{model_depth}'

    if rank is not None:
        extension += f'_{rank}'
    program_path = Path(__file__)
    fname = str(program_path.parent.absolute()) + extension
    return fname


def count_frequency(fname=None, binarised=False,
                    perm_inv=False, mask=None, return_unq=False, matrix=None):
    # Gets  data from file, changes it into 2D numpy tensor and effectively
    # counts how often each row or 'function' occurs
    if matrix is None:
        tensor = np.loadtxt(fname=fname, delimiter=',')
    else:
        tensor = matrix

    if binarised:
        tensor = tensor % 2

    if perm_inv:
        tensor = permutational_order(tensor)
    print(tensor.shape)

    if mask is not None:
        tensor = tensor[:, mask]  # Applies the mask to each row

    print(tensor.shape)

    unq, cnt = np.unique(tensor, return_counts=True, axis=0)
    print(unq)
    if return_unq:
        return unq, cnt
    else:
        return np.sort(cnt)[::-1]


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


def test_accuracy(model, data, epoch_num=None, on_training_data=True):
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
        acc = int(correct) / test_num
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
        err[0] = probability - z * np.sqrt(err[0] * (1 - err[0]) /
                                           test_size)
        err[1] = probability + z * np.sqrt(err[1] * (1 - err[1]) /
                                           test_size)

    return err

def theoretical_val(func_len, output_len, cut_off_rank=1, p_vio=0, classes=2):
    C = (1-p_vio)/(func_len*math.log(classes)-math.log(cut_off_rank))

    x = np.arange(1, output_len+1)
    y = C/x
    print(x)
    print(C)
    return x, y


def produce_rankVProb_plot(*arrays, labels=None,
                           title="Rank vs Probability",
                           xlabel="Rank",
                           ylabel="Probability",
                           log_scale=True,
                           cumulative=False,
                           error=False,
                           theoretical=False, function_length=None,
                           fname=None):
    max_length = 0
    arrays = list(arrays)

    test_size = np.zeros(len(arrays))
    for i in range(len(arrays)):
        max_length = max(max_length, arrays[i].size)
        test_size[i] = np.sum(arrays[i])
        arrays[i] = 1 / np.sum(arrays[i]) * arrays[i]  # Normalises array
        print(np.sum(arrays[i]))

    print("THIS")
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_box_aspect(1)
    ax.set_xlim([1, 1e8])
    ax.set_ylim([1e-8, 1])

    for i in range(len(arrays)):
        # Makes sure all the arrays are the same length
        # l = max_length-arrays[i].size
        # arrays[i] = np.concatenate((arrays[i], np.zeros(l)))
        rank = np.arange(1, arrays[i].size + 1)
        if error:
            e2 = calc_row_err(arrays[i], test_size[i], depth=5)
            plt.fill_between(rank, e2[0], e2[1],
                             alpha=0.3)
        if cumulative:
            arrays[i] = np.cumsum(arrays[i])

        if labels:
            p = plt.plot(rank, arrays[i], label=labels[i])
            plt.legend()
        else:
            p = plt.plot(rank, arrays[i])
            print('hehe')

        if theoretical:
            if function_length==None:
                raise TypeError("Function length isn't specified...")
            c = 30
            x_fit, y_fit = theoretical_val(func_len=function_length,
                                        output_len=len(
                arrays[i]), p_vio=arrays[i][:c].sum(), cut_off_rank=c+1)
            plt.plot(x_fit, y_fit, linestyle='dotted', color=p[0].get_color())

            print(f'Violating probabilities '
                  f'{arrays[i][:2]}, {arrays[i][:2].sum()}')

        plt.xlabel('Rank')
        plt.ylabel('Probability')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    if fname:
        plt.savefig(fname, bbox_inches='tight')
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


def reduced_mask(dataset_name, group_size, org_group_size=20):
    # This produces a mask that takes in a function which has size of
    # org_group_size*num_classes and reduces it so that its now
    # group_size*num_classes, i.e. just reduces size of function so that in
    # our new function it has a certain amount of each function
    dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    print('Getting reduced mask :)')
    data = dataset[0]
    num_classes = dataset.num_classes
    # # This is the mask with 20 lots of each class which is the original amount
    m1 = generate_mask(data_y=data.y, group_size=org_group_size,
                       num_classes=num_classes,
                       mask=data.train_mask, name=dataset_name).numpy()

    # This is subset of original mask
    m2 = generate_mask(data_y=data.y, group_size=group_size,
                       num_classes=num_classes,
                       mask=m1, name=dataset_name).numpy()
    print(np.unique(data.y[m2].numpy(), return_counts=True))
    return m2[m1]

def bring_together_file(dataset_name, train_it, model_type, model_depth,
                        prefix=None, del_it=False, save_it=True):
    # FINISH THIS PART OF THE CODE
    program_path = Path(__file__)
    if prefix is None:
        prefix = '/output/'

    path_to_output = str(program_path.parent.absolute()) + prefix

    print(path_to_output)
    # Gets a list of filenames all with the same rank
    file_names = [filename for filename in listdir(path_to_output) if
             filename.startswith(f'{dataset_name}'
                                 f'_{"trained" if train_it else "random"}_'
                                 f'{model_type}_{model_depth}')]
    print('Ok bringing together')
    print(file_names)

    files = sorted([path_to_output+f1 for f1 in file_names])
    combined_txt = ""
    for file in files:
        with open(file, 'r') as f:
            combined_txt += f.read()
    # print(combined_txt)
    # print((len(combined_txt))/240)
    fname = get_file_name(dataset_name, train_it, model_type, model_depth,
                          prefix=prefix)
    print(fname)
    print(len(combined_txt)/240)
    # # write the combined text to a new file
    # Add a section here that asks you to confirm before you send it off if
    # it already exists (prevents multiple writes to the same file)

    if exists(fname):
        if input('Enter y if you want to continue, this file already '
                 'exists... ') != 'y':
            return None

    if save_it:
        with open(fname, 'a') as f:
            f.write(combined_txt)

    if del_it:
        for file in files:
            os.remove(file)

# bring_together_file('CiteSeer', False, 'GfNN', 6, prefix=pre, del_it=False,
#                     save_it=True)

def wrap_it_all_up_Cite(train_it, model_type, depth, prefix='/bring/'):

    fname1 = get_file_name("CiteSeer", train_it, model_type, depth,
                           prefix=prefix)
    freq1 = count_frequency(fname1, binarised=True)
    np.save(arr=freq1, file=get_file_name("CiteSeer", train_it, model_type,
                                          depth, rank=120, prefix='/freq/'))

    # produce_rankVProb_plot(freq1, error=True)

# wrap_it_all_up_Cite(False, 'GCN', 2)
# wrap_it_all_up_Cite(False, 'GfNN', 2)
# wrap_it_all_up_Cite(False, 'GCN', 6)
# wrap_it_all_up_Cite(False, 'GfNN', 6)


train_it = False
depth = 6
model = 'GCN'
f1 = get_file_name('CiteSeer', train_it, model, depth, prefix='/freq/')
f2 = get_file_name('CiteSeer', train_it, model, depth, prefix='/plots/')
freq1 = np.load(f1 + ".npy")
produce_rankVProb_plot(freq1, theoretical=True, function_length=24, labels=[
    f'{model}, depth: {depth}, function length: 24'], fname=f2)

# count_frequency(fname1)
# fname2 = get_file_name("Cora", True, "GCN", 6)



# CODE FOR PRODUCING PROB V PROB PLOT, ENTER INTO FUNCTION
# unq1, freq1 = count_frequency(fname1, mask=reduced_mask(3), binarised=True)
# unq2, freq2 = count_frequency(fname2, mask=red_mask_for_cora(3), binarised=True)
#
# unq1 = ["".join([str(int(j)) for j in i]) for i in unq1]
# unq2 = ["".join([str(int(j)) for j in i]) for i in unq2]
# total1 = int(np.sum(freq1))
# total2 = int(np.sum(freq2))
# df1 = pd.DataFrame.from_dict({'f':unq1, 'c':freq1/total1})
# df1.sort_values(by=['c'], inplace=True, ascending=False)
# df1.reset_index(drop=True, inplace=True)
# df2 = pd.DataFrame.from_dict({'f':unq2, 'c':freq2/total2})
# df2.sort_values(by=['c'], inplace=True, ascending=False)
# df2.reset_index(drop=True, inplace=True)
#
# df=pd.merge(df1,df2,left_on='f',right_on='f',how='outer',indicator=True)
#
# df['c_x'] = [i if not math.isnan(i) else 1/total1 for i in df['c_x']]
# df['c_y'] = [i if not math.isnan(i) else 1/total2 for i in df['c_y']]
# print(df)
#
# fig, ax = plt.subplots()
# ax.scatter(df['c_x'], df['c_y'])
# ax.set_xscale('log')
# ax.set_yscale('log')
# plt.show()

# freq1 = count_frequency(fname=fname1, mask=reduced_mask("CiteSeer", 5),
#                         binarised=True)
# produce_rankVProb_plot(freq1)


# print(list(sm))
# print("-------------------------")
# print("data after")
# print(list(nsm))
# print(data.x[data.train_mask][0])
