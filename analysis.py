import numpy as np
import torch
from data_handler import get_file_name, count_same_p, produce_rankVProb_plot,\
    produce_probVprob, wrap_it_all_up, count_frequency, bring_together_file
import data_handler
import matplotlib.pyplot as plt

# data=torch.load('CiteSeer')
# x, y = data.edge_index
# plt.scatter(x, y)
# plt.show()

def produce_final_rankVprob(*args, dataset_name, freq_prefix, num_classes,
                            plots_prefix=None, error=False):

    # Depth function length and group_size are parameters
    # Takes in a series of tuples representing what we want to plot e.g. (
    # 'GCN', 2, 4), ('GfNN, 6, 4) for plotting two graphs with group size 4
    # and depths 2 and 6 and both models
    args = list(args)
    freq = []


    labels = [""] * len(args)
    func_lens = [0]*len(args)

    for i, x in enumerate(args):
        model_type, model_depth, group_size = x
        f1 = data_handler.get_file_name(dataset_name, False, model_type,
                                        model_depth, prefix=freq_prefix,
                                        rank=group_size)
        freq.append(np.load(f1+".npy"))
        labels[i] = f'{model_type}, depth: {model_depth}, ' \
                    f'function length {group_size * num_classes}'
        func_lens[i] = group_size * num_classes

    fname = get_file_name(dataset_name, False, model_type=args[0][0],
                          model_depth=0, prefix=plots_prefix) + str(args)[1:-1]
    print(fname)
    produce_rankVProb_plot(*freq, labels=labels,
                        function_length=func_lens,
                        fname=fname, theoretical=True, error=error)


# wrap_it_all_up('CiteSeer', False, 'GCN', 6, '/output_CiteSeer/',
#                '/output_final2/', '/freq2/', group_size=20)
# wrap_it_all_up('CiteSeer', False, 'GCN', 2, '/output_CiteSeer/',
#                '/output_final2/', '/freq2/', group_size=20)
# wrap_it_all_up('CiteSeer', False, 'GfNN', 2, '/output_CiteSeer/',
#                '/output_final2/', '/freq2/', group_size=20)
# wrap_it_all_up('CiteSeer', False, 'GfNN', 6, '/output_CiteSeer/',
#                '/output_final2/', '/freq2/', group_size=20)


produce_final_rankVprob(('GCN', 2, 4), ('GCN', 2, 20),
                        dataset_name='CiteSeer', freq_prefix='/freq2/',
                        num_classes=6, plots_prefix='/plots2/')




def produce_final_probVprob(model_depth, dataset_name,
                            group_size,
                            input_prefix, plots_prefix=None, binarised=True):

    if plots_prefix == None:
        fname = None
    else:
        fname = get_file_name(dataset_name, True, 'both', model_depth,
                              prefix=plots_prefix, rank=group_size)

    data = torch.load(dataset_name)
    mask= data_handler.reduced_mask(dataset_name, group_size=group_size,
                                    data=data)

    z, unq = count_same_p(dataset_name, model_depth, input_prefix, mask=mask,
                          binarised=binarised)

    x, y = z
    # Needed to get the accuracy of each point
    true_data = data.y[data.test_mask][mask]

    produce_probVprob(x, y, fname=fname, s=8, unq=unq, true_data=true_data,
                      label=['hey'], theoretical=True, binarised=binarised)


# bring_together_file('Synth', True, 'GCN', 6, f_prefix='/output_CiteSeer/',
#                     output_prefix='/output_final2/')

# produce_final_probVprob(model_depth=6, dataset_name='Synth', group_size=20,
#                         input_prefix='/output_final2/', binarised=False,
                        # plots_prefix='/plots2/')


# plots_prefix='/plots2/'
# produce_final_probVprob(2, 'Synth', 12, '/output_final2/',
#                         plots_prefix='/plots2/')
