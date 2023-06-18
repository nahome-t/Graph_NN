# This file generates the commands that you can enter into the hydra
# computer, in order to insure the program runs make sure there is a local
# folder called output

def generate_text(dataset_name, train_it, processes, total, model_type, model_depth,
                  path_to_code=None, min_processes=0):

    test_num = int(total/processes)
    for i in range(min_processes, min_processes+processes):
        print(f'python3 {path_to_code} --test_num {test_num} --train_it'
              f' {train_it} '
              f'--model_type {model_type} --model_depth {model_depth} --rank '
              f'{i} --dataset_name {dataset_name}')



def gfNN_tests():
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    untrained_tests = 10 ** 7
    trained_tests = 10 ** 5
    generate_text('CiteSeer', train_it=False, processes=25,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=25,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)

    generate_text('CiteSeer', train_it=True, processes=25,
                  total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)

    generate_text('CiteSeer', train_it=True, processes=25,
                  total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)


def ddd():

    untrained_tests= 10**7
    trained_tests = 10**5
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'


    generate_text('CiteSeer', train_it=True, processes=5, total=trained_tests,
                  model_type= 'GCN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=5, total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=40, total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=True, processes=40, total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)


    generate_text('CiteSeer', train_it=False, processes=5,
                  total=untrained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=5,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=5,
                  total=untrained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=5,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)

def dd():
    untrained_tests = 10**7
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    generate_text('CiteSeer', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p)
    generate_text('CiteSeer', train_it=False, processes=20,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p)

def d():
    unit = 500000
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'
    generate_text('CiteSeer', train_it=False, processes=60,
                  total=13*unit,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=100)

    generate_text('CiteSeer', train_it=False, processes=20,
                  total=4 * unit,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=100)

def s():
    tests = 600000
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'
    generate_text('CiteSeer', train_it=False, processes=50,
                  total=tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=200)
# ddd()

# generate_text('CiteSeer', train_it=False, processes=1, total=1,
#               model_type='GCN', model_depth=6,
#               path_to_code='/mnt/zfsusers/nahomet/Graph_NN/runner.py')
def sanity():
    tests = 1000000
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    generate_text('CiteSeer', train_it=False, processes=100,
                  total=tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=0)



def final_untrained():
    tests = 10**7
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    generate_text('CiteSeer', train_it=False, processes=25,
                  total=tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=0)
    generate_text('CiteSeer', train_it=False, processes=25,
                  total=tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=0)


    generate_text('CiteSeer', train_it=False, processes=25,
                  total=tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=0)
    generate_text('CiteSeer', train_it=False, processes=25,
                  total=tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=0)


def final_trained(small):
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'
    trained_tests = 10**5

    generate_text('CiteSeer', train_it=True, processes=20, total=trained_tests,
                  model_type= 'GCN', model_depth=2,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=True, processes=20, total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=True, processes=50, total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=True, processes=50, total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=small)

def sp_final_trained(small):
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'
    trained_tests = 10 ** 5
    untrained_tests = 10**7

    generate_text('CiteSeer', train_it=True, processes=10, total=50000,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=True, processes=2, total=4000,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=True, processes=60, total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=small)
    generate_text('CiteSeer', train_it=False, processes=25, total=4*10**6,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=small)


def extra_untrained():
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'
    tests = 10**7

    generate_text('CiteSeer', train_it=False, processes=20,
                  total=tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=100)
    generate_text('CiteSeer', train_it=False, processes=20,
                  total=tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=100)


    generate_text('CiteSeer', train_it=False, processes=20,
                  total=tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=100)
    generate_text('CiteSeer', train_it=False, processes=20,
                  total=tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=100)

def gen_wrap_text(model, depth, train_it):
    p = '/mnt/zfsusers/nahomet/Graph_NN/data_handler.py'
    s = f'python3 {p} --model_type {model} --model_depth {depth} ' \
         f'--train_it {train_it} ' \
 '--group_size 4 --output_prefix /output/ --freq_prefix /freq/'
    print(s)


def synthetic():
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    untrained_tests = 10**7
    trained_tests = 10**5
    sc1 = 30
    sc2 = 20
    generate_text('Synth', train_it=True, processes=sc1,
                  total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=True, processes=sc1,
                  total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=0)

    generate_text('Synth', train_it=True, processes=sc2,
                  total=trained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=True, processes=sc2,
                  total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=0)



    generate_text('Synth', train_it=False, processes=sc1,
                  total=untrained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=False, processes=sc1,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=0)

    generate_text('Synth', train_it=False, processes=sc2,
                  total=untrained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=False, processes=sc2,
                  total=untrained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=0)


def synthetic_trained():
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

    trained_tests = 10**5
    generate_text('Synth', train_it=True, processes=15,
                  total=trained_tests,
                  model_type='GCN', model_depth=6,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=True, processes=15,
                  total=trained_tests,
                  model_type='GfNN', model_depth=6,
                  path_to_code=p, min_processes=0)

    generate_text('Synth', train_it=True, processes=10,
                  total=trained_tests,
                  model_type='GCN', model_depth=2,
                  path_to_code=p, min_processes=0)
    generate_text('Synth', train_it=True, processes=10,
                  total=trained_tests,
                  model_type='GfNN', model_depth=2,
                  path_to_code=p, min_processes=0)


def CiteSeer_all():

    untrained_tests= 10**7
    trained_tests = 10**5
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'


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



def synth_all():
    untrained_tests= 10**7
    trained_tests = 10**5
    p = '/mnt/zfsusers/nahomet/Graph_NN/runner.py'

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

