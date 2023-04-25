import numpy as np
import torch
from data_handler import get_file_name, count_same_p, produce_rankVProb_plot,\
    produce_probVprob, wrap_it_all_up
import data_handler

def produce_final_rankVprob(*args, dataset_name, freq_prefix, num_classes, plots_prefix=None):

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

    produce_rankVProb_plot(*freq, labels=labels,
                                        function_length=func_lens,
                                        fname=plots_prefix, theoretical=True)


# wrap_it_all_up('CiteSeer', False, 'GCN', 6, '/output_CiteSeer/',
#                '/output_final2/', '/freq2/', group_size=4)






def produce_final_probVprob(model_depth, dataset_name,
                            group_size, org_group_size,
                            output_prefix, fname=None):



    data = torch.load(dataset_name)
    mask= data_handler.reduced_mask(dataset_name, group_size=group_size,
                                    data=data)

    z, unq = data_handler.count_same_p(dataset_name, model_depth,
                                       output_prefix, mask=mask)

    x, y = z
    # Needed to get the accuracy of each point
    true_data = data.y[data.test_mask][mask]

    produce_probVprob(x, y, fname=fname, s=8, unq=unq, true_data=true_data,
                      label=['hey'], theoretical=True)



