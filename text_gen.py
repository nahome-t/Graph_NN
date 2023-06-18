# This file generates the text for the commands that you can enter into
# hydra, this is then copied into a params.txt file which should be saved
# within the Graph_NN folder instructions on executing this file found here
# https://www2.physics.ox.ac.uk/it-services/multirun-running-serial-programs-in-parallel-on-the-astro-or-theory-hpc-clusters

def generate_text(dataset_name, train_it, processes, total, model_type, model_depth,
                  path_to_code=None, min_processes=0):

    # processes is the number of cores the program is sent to, the output for
    # each is saved with a prefix _rank, where here the rank ranges from
    # [min_processes, min_processes + processes]
    test_num = int(total/processes)
    for i in range(min_processes, min_processes+processes):
        print(f'python3 {path_to_code} --test_num {test_num} --train_it'
              f' {train_it} '
              f'--model_type {model_type} --model_depth {model_depth} --rank '
              f'{i} --dataset_name {dataset_name}')




def CiteSeer_all(p='/mnt/zfsusers/nahomet/Graph_NN/runner.py'):

    untrained_tests= 10**7
    trained_tests = 10**5

    generate_text('CiteSeer', train_it=True, processes=5, total=trained_tests,
                  model_type= 'GCN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=5, total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=50, total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=40, total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)


    generate_text('CiteSeer', train_it=False, processes=50,
                  total=untrained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=25,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=80,
                  total=untrained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=40,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)



def synth_all(p='/mnt/zfsusers/nahomet/Graph_NN/runner.py'):
    untrained_tests= 10**7
    trained_tests = 10**5

    generate_text('Synth', train_it=False, processes=8,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=80,
                  total=untrained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)



    generate_text('Synth', train_it=False, processes=4,
                  total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=25,
                  total=trained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=8,
                  total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)

    generate_text('Synth', train_it=False, processes=25,
                  total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)

