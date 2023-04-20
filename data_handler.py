# Contains functions used to read and write data
import argparse
import os
from pathlib import Path
import pandas as pd
# import scipy.optimize
import torch, math
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
from models import NormAdj
from torch_geometric.datasets import Planetoid
from os import listdir
# from scipy.optimize import curve_fit

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
                  prefix='/output/', dir=False):
    # Gets the file name that an output should be saved to given the name of
    # a dataset, whether its trained or not and the model type or depth

    extension = f'{prefix}{dataset_name}' \
                f'_{"trained" if train_it else "random"}_' \
                f'{model_type}_{model_depth}'

    if rank is not None:
        extension += f'_{rank}'
    program_path = Path(__file__)
    if dir:
        return str(program_path.parent.absolute())
    fname = str(program_path.parent.absolute()) + extension
    return fname


def count_frequency(fname=None, binarised=False,
                    perm_inv=False, mask=None, return_unq=False, matrix=None,
                    special=None):
    # Gets  data from file, changes it into 2D numpy tensor and effectively
    # counts how often each row or 'function' occurs
    if matrix is None:
        if special:
            tensor = np.loadtxt(fname=fname, delimiter=',', max_rows=special)
        else:
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

    return acc



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


def theoretical_val(func_len, output_len, cut_off_rank=1, p_vio=0, classes=2,
                    alpha=-1):

    C = (1 - p_vio) / (func_len * math.log(classes) - math.log(cut_off_rank))

    x = np.arange(1, output_len + 1)
    y = C / x
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
                           fname=None,
                           extra=None):

    arrays = list(arrays)

    test_size = np.zeros(len(arrays))
    for i in range(len(arrays)):

        test_size[i] = np.sum(arrays[i])
        arrays[i] = 1 / np.sum(arrays[i]) * arrays[i]  # Normalises array
    arrays.append(extra)

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
            if function_length == None:
                raise TypeError("Function length isn't specified...")
            c = 30
            x_fit, y_fit = theoretical_val(func_len=function_length,
                                           output_len=len(
                                               arrays[i]),
                                           p_vio=arrays[i][:c].sum(),
                                           cut_off_rank=c + 1)
            plt.plot(x_fit, y_fit, linestyle='dotted', color=p[0].get_color())

            print(f'Violating probabilities '
                  f'{arrays[i][:2]}, {arrays[i][:2].sum()}')

        plt.xlabel('Rank')
        plt.ylabel('Probability')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
    if fname:
        fig.set_size_inches(3, 3)
        plt.savefig(fname, bbox_inches='tight', dpi=300)

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
                        f_prefix, output_prefix, del_it=False, save_it=True,
                        rank=None):
    # FINISH THIS PART OF THE CODE
    program_path = Path(__file__)
    if f_prefix is None:
        f_prefix = '/output/'

    path_to_output = str(program_path.parent.absolute()) + f_prefix

    print(path_to_output)
    starting_with = f'{dataset_name}_{"trained" if train_it else "random"}_' \
                    f'{model_type}_{model_depth}'
    # Gets a list of filenames all with the same rank
    file_names = [filename for filename in listdir(path_to_output) if
                  filename.startswith(starting_with)]
    print(f'Ok bringing together starting with: {starting_with}')
    print(file_names)

    files = sorted([path_to_output + f1 for f1 in file_names])
    combined_txt = ""
    for file in files:
        with open(file, 'r') as f:
            combined_txt += f.read()
    # print(combined_txt)
    # print((len(combined_txt))/240)
    output_fname = get_file_name(dataset_name, train_it, model_type,
                                 model_depth,
                                 prefix=output_prefix, rank=rank)
    print(output_fname)
    print(len(combined_txt) / 240)
    # # write the combined text to a new file
    # Add a section here that asks you to confirm before you send it off if
    # it already exists (prevents multiple writes to the same file)

    if exists(output_fname):
        if input('Enter y if you want to continue, this file already '
                 'exists... ') != 'y':
            return None

    if save_it:
        with open(output_fname, 'a') as f:
            f.write(combined_txt)

    if del_it:
        for file in files:
            os.remove(file)


def wrap_it_all_up_Cite(train_it, model_type, model_depth,
                        output_prefix=None,
                        freq_prefix=None, group_size=None):
    if output_prefix is None:
        output_prefix = '/output'

    if freq_prefix is None:
        freq_prefix = '/freq/'
    if group_size is None:
        print('No group size given, assuming no mask needed')

    # Brings together file and outputs it in the same spot as all the outputs
    bring_together_file('CiteSeer', train_it, model_type, model_depth,
                        f_prefix=output_prefix, output_prefix=output_prefix)
    print('ggg')
    # Gets this file
    combined_file = get_file_name("CiteSeer", train_it, model_type, model_depth,
                                  prefix=output_prefix)
    # Counts frequency
    freq1 = count_frequency(combined_file, binarised=True, mask=reduced_mask(
        'CiteSeer', group_size=group_size))

    np.save(arr=freq1, file=get_file_name("CiteSeer", train_it, model_type,
                                          model_depth, rank=group_size,
                                          prefix=freq_prefix))


# wrap_it_all_up_Cite(False, 'GfNN', 2, output_prefix='/output5/output/',
#                     rank=None)

# wrap_it_all_up_Cite(train_it=False, model_type='GCN', model_depth=2,
#                     output_prefix='/output_final_hopefully/output/',
#                     freq_prefix='/output_final_hopefully/freq/', group_size=4)

# wrap_it_all_up_Cite(train_it=True, model_type='GCN', model_depth=2,
#                     output_prefix='/output_final_hopefully/output/',
#                     freq_prefix='/output_final_hopefully/freq/', group_size=4)

# train_it = False
# depth = 6
# model = 'GCN'
# #
# f1 = get_file_name('CiteSeer', train_it, model, depth, prefix=
# '/output_final_hopefully/freq/')
# freq1 = np.load(f1 + ".npy")
# f2 = get_file_name('CiteSeer_OFF', train_it, model, depth, prefix='/plots/')
# produce_rankVProb_plot(freq1, theoretical=True, function_length=24, labels=[
#     f'{model}, depth: {depth}, function length: 24'])


# parser2 = argparse.ArgumentParser(
#     description='Auto running wrap it all up Cite, where it wraps it all up '
#                 'for the CiteSeer tests and produces our frequency vs '
#                 'probability plot')
# parser2.add_argument('--model_type', type=str, help='GCN or GfNN')
# parser2.add_argument('--train_it', type=str, help='True or False')
# parser2.add_argument('--model_depth', type=int, help='Depth of the of the '
#                                                      'neural network model')
# parser2.add_argument('--output_prefix', type=str,
#                      help='What folder the output is stored in e.g. "/output/" '
#                           'by default')
# parser2.add_argument('--freq_prefix', type=str,
#                      help='What folder the frequency data is stored in e.g. '
#                           '"/freq/" by default')
# parser2.add_argument('--rank', type=int,
#                      help='Helps with multiprocessing, writes to new filename '
#                           'which is "fname_rank", be wary of using this as '
#                           'files normally saved with rank between 1-1000 when '
#                           'multiprocessing')
# parser2.add_argument('--group_size', type=int,
#                      help='How much of each class do you want in function')
# args2 = parser2.parse_args()
#
# if args2.model_type is not None:
#     train_it = args2.train_it
#     if train_it not in {'False', 'True'}:
#         raise ValueError('Not a valid boolean string')
#     train_it = train_it == 'True'
#     print(args2)
#
#     wrap_it_all_up_Cite(train_it=train_it, model_type=args2.model_type,
#                         model_depth=args2.model_depth,
#                         output_prefix=args2.output_prefix,
#                         freq_prefix=args2.freq_prefix, rank=args2.rank,
#                         group_size=args2.group_size)

def produce_probVprob(x, y, fname=None, log_scale=True, title=None,
                      label=None, s=None):
    # Produces probablility vs prbability plot for same model but comparing
    # trained and untrained

    _, ax = plt.subplots()
    ax.set_box_aspect(1)

    plt.xlim([10 ** (-5.5), 1])
    plt.ylim([10 ** (-5.5), 1])
    if label:
        plt.scatter(x, y, label="lol")
        plt.legend()
    else:
        if s is not None:
            plt.scatter(x, y, s=s)
        else:
            plt.scatter(x, y)

    if title is not None:
        plt.title(title)

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')

    plt.xlabel('P(f|T) for GCN')
    plt.ylabel('P(f|T) for GfNN')

    # ax.set_xlim([1, 10**5])
    # ax.set_ylim([10**(-5), 1])

    if fname:
        plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()


def count_same_p(dataset_name, model_depth, prefix,
                 group_size, freq_prefix=None):
    if freq_prefix is None:
        print('you havent written frequnency prefix, this wont get saved')
    fname1 = get_file_name(dataset_name, True, 'GCN', model_depth,
                           prefix=prefix)
    fname2 = get_file_name(dataset_name, True, 'GfNN', model_depth,
                           prefix=prefix)
    unq1, freq1 = count_frequency(fname1, mask=reduced_mask(dataset_name,
                                                            group_size),
                                  binarised=True, return_unq=True,
                                  special=10 ** 5)
    unq2, freq2 = count_frequency(fname2, mask=reduced_mask(dataset_name,
                                                            group_size),
                                  binarised=True, return_unq=True,
                                  special=10 ** 5)
    print(freq1.shape)
    print(freq2.shape)

    unq1 = ["".join([str(int(j)) for j in i]) for i in unq1]
    unq2 = ["".join([str(int(j)) for j in i]) for i in unq2]
    total1 = int(np.sum(freq1))
    total2 = int(np.sum(freq2))
    df1 = pd.DataFrame.from_dict({'f': unq1, 'c': freq1 / total1})
    df1.sort_values(by=['c'], inplace=True, ascending=False)
    df1.reset_index(drop=True, inplace=True)
    df2 = pd.DataFrame.from_dict({'f': unq2, 'c': freq2 / total2})
    df2.sort_values(by=['c'], inplace=True, ascending=False)
    df2.reset_index(drop=True, inplace=True)

    df = pd.merge(df1, df2, left_on='f', right_on='f', how='outer',
                  indicator=True)

    df['c_x'] = [i if not math.isnan(i) else 1 / total1 for i in df['c_x']]
    df['c_y'] = [i if not math.isnan(i) else 1 / total2 for i in df['c_y']]

    res = np.array([df['c_x'], df['c_y']])

    # if freq_prefix:
    #     np.save(get_file_name(dataset_name, train_it='both',
    #                           model_type='both',
    #                                       model_depth=model_depth, rank=group_size,
    #                                       prefix=freq_prefix), res)
    return res


# output_prefix = '/output_final/'
# dataset_name = 'CiteSeer'
# model_depth = 6
# group_size = 6
# freq_prefix = '/freq/'

# fname1 = get_file_name(dataset_name, True, 'GCN', model_depth,
#                        prefix=output_prefix)
# f1 = count_frequency(fname1, binarised=True, mask=reduced_mask('CiteSeer', 4))
# produce_rankVProb_plot(f1)

# print(get_file_name(dataset_name, train_it='both',
#                               model_type=model_type,
#                                           model_depth=model_depth, rank=4,
#                                           prefix='/freq/'))

# z = count_same_p('CiteSeer', model_depth, output_prefix, group_size=group_size,
#                   freq_prefix=freq_prefix)
# fname = get_file_name('CiteSeer', True, 'both', model_depth, prefix='/plots/')
# print(fname)
# print(z.shape)
# x, y = z
# produce_probVprob(x, y, fname=fname, s=8)


# # --------------------------------------------------------------------------- #
# fname = get_file_name('CiteSeer', train_it=False, model_type='GCN', \
#     model_depth=2, prefix='/output_final_hopefully/freq/') + ".npy"

# fname2 = get_file_name('CiteSeer', train_it=False, model_type='GCN', \
#     model_depth=6,
#                       prefix='/output_final/')

# freq2 = count_frequency(fname2, perm_inv=True, special=10*10**5,
#                         mask=reduced_mask('CiteSeer', 4))


# freq1 = np.load(fname)
#
# total = np.sum(freq1)
# p1 = freq1/total
# r_min = 70
# l = 24
# p_v = np.sum(p1[:r_min])
#
# sigma = np.sqrt(p1*(1-p1)/10**7)[r_min:]
#
# def model(x, alpha):
#     a = 1-alpha
#     C = a*(1-p_v)/(2**(l*a) - r_min**(a))
#     output = C/(np.power(x, alpha))
#     return output


# -----------------------------------------------------------------------#
# code for fitt
# # bounds = ([0.5, 10**-3], [0.95, 10**-1])
# bounds = ([0.5], [0.95])
# popt, pcov = curve_fit(model, xdata=np.arange(r_min, len(freq1)),
#                        ydata=p1[r_min:], bounds=bounds, sigma=sigma)
# print(np.sqrt(pcov))
# print(popt)
# rank = np.arange(1, len(freq1) +1)
# produce_rankVProb_plot(freq1, extra=model(rank, popt[0]))
