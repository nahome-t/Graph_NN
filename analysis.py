import numpy as np

import data_handler

def produce_final_rankVprob(*args, dataset_name, freq_prefix, num_classes, plots_prefix=None):

    # Depth function length and group_size are parameters
    # Takes in a series of tuples representing what we want to plot e.g. (
    # 'GCN', 2, 24)
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

    data_handler.produce_rankVProb_plot(*freq, labels=labels,
                                        function_length=func_lens,
                                        fname=plots_prefix, theoretical=True)


produce_final_rankVprob(('GCN', 2, 4), ('GFNN', 2, 4),
                        dataset_name='CiteSeer',
                        freq_prefix='/output_final_hopefully/freq/',
                        num_classes=6)


    # Code for producing final form of plots
    # model_depth=6
    # group_size=120
    # f1 = get_file_name('Synth', train_it=False,
    #                             model_type='GCN',
    #                             model_depth=model_depth,
    #                             prefix='/freq/', rank=group_size)
    # f2 = get_file_name('Synth', train_it=False,
    #                             model_type='GfNN',
    #                             model_depth=model_depth,
    #                             prefix='/freq/', rank=group_size)
    #
    # freq1 = np.load(f1+".npy")
    # freq2 = np.load(f2+".npy")
    # group_size=20
    #
    # # f3 = count_frequency(fname=)
    # basis = f'GCN, depth {model_depth}, function length: {group_size*6}'
    # basis2 = f'GfNN, depth {model_depth}, function length: {group_size*6}'
    #
    # produce_rankVProb_plot(freq1, freq2,
    #                        labels=[basis, basis2],
    #                        theoretical=True, function_length=[group_size*6,
    #                                                           group_size*6],
    #                        c=40)
    #
    # print(get_file_name(dataset_name, train_it='both',
    #                               model_type=model_type,
    #                                           model_depth=model_depth, rank=4,
    #                                           prefix='/freq/'))





def produce_final_probVprob(model_type, model_depth, group_size=4, ):


    # Code for how i ran it once
    # group_size = 4
    # dataset_name = 'CiteSeer'
    # num_classes = 6
    # org_group_size = 20
    #
    #
    # fname = get_file_name(dataset_name, True, 'both', model_depth, prefix='/plots/')
    # if dataset_name == 'Synth':
    #     data = torch.load('synth3')
    #     print('loading synth')
    # else:
    #     print('loading cora')
    #     dataset = Planetoid(root=f'/tmp/{dataset_name}', name=dataset_name)
    #     data = dataset[0]
    #
    # mask=reduced_mask(dataset_name, group_size=group_size, org_group_size=org_group_size,
    #                   data=data, num_classes=num_classes)
    #
    # z, unq = count_same_p(dataset_name, model_depth, output_prefix, mask=mask)
    #
    #
    # # mask = generate_mask(data_y=data.y, )
    #
    # x, y = z
    #
    # # CHANGE THIS FOR CITESEER !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # true_data = data.y[data.test_mask][mask]
    #
    # produce_probVprob(x, y, fname=fname, s=8, unq=unq, true_data=true_data,
    #                   label=['hey'], theoretical=True)

    return None
