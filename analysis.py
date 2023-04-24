import data_handler

def produce_final_rankVprob():
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


    return None



def produce_final_probVprob():
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
