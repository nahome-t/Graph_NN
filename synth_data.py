import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch import from_numpy
from torch_geometric.utils import to_undirected


from data_handler import generate_mask, get_file_name

# variables

print(np.version.version)


def mapping(r, eta, scale):
    r_n = r*math.pi/scale
    s = np.sin(r_n)
    return 0.5*s/np.sqrt(s**2 + eta) +0.5

def mapping2(r, k, scale):
    return 1/(1+np.exp(k*(r-scale)))

def prob_to_bin(arr):
    return 1-(np.random.random(size=arr.size)>arr).astype(int)



def plot_it(plot_range, class_var_scale, X, Y, edge_index=None, map=mapping,
            eta=0.2, fname=None):
    grid_res = 1000 # How many intervals plotted for x axis

    # This defines our mapping from positions to probabilities (values here will
    # range from 0 to approximately 1, note choose eta<0.1 for wave effect
    p_x, p_y = X.T

    x, y = np.meshgrid(np.linspace(-plot_range, plot_range, grid_res), np.linspace(
        -plot_range, plot_range, grid_res))
    r = np.sqrt(x*x + y*y)
    z = map(r, eta, scale=class_var_scale)

    fig, ax = plt.subplots()

    ax.scatter(p_x, p_y, zorder=2, s=15, c=Y, edgecolors='black')
    c = ax.pcolormesh(x, y, z, cmap='RdYlBu_r', vmin=0, vmax=1, zorder=-1)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    cbar = fig.colorbar(c, ax=ax, shrink=0.9)
    cbar.ax.set_ylabel('Probability of being in class 1')
    ax.set_box_aspect(1)

    if edge_index is not None:
        for pair in edge_index.T:
            print(pair)
            ax.plot(p_x[pair], p_y[pair], '-o', zorder=1, color='black')

    if fname is not None:
        fig.set_size_inches(5, 5)
        plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.show()


def generate_data(n_points, pos_scale, class_var_scale, eta = 0.1,
                  near_neighbours
=4, ret_data=True, train_size=None, map=mapping, dim=2, noise=0.05):

    X = np.random.normal(scale=pos_scale, size=(n_points, dim))

    P_r = np.linalg.norm(X, axis=-1)

    P = map(P_r, eta, class_var_scale)  # Probability of each point being
    # class 1
    # An actual value assigned to each node (either 0 or 1 corresponding to
    # probability

    Y = prob_to_bin(P)
    if noise is not None:
        X = np.random.normal(scale=noise, size=(n_points, dim)) + X

    nbrs = NearestNeighbors(n_neighbors=near_neighbours, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Currently index in adjacency list format, this converts it to edge
    # index format [i1, i2, i3, ...], [j1, j2, j3, ...]

    dim1, dim2 = indices.shape
    s1 = dim1*(dim2-1)
    edge_index = np.zeros((2, s1), dtype=int)
    edge_index[0] = np.floor_divide(np.arange(s1), near_neighbours-1)
    edge_index[1] = indices[:, 1:].reshape(-1)
    # Unique only gets the pairs that appear once
    edge_index = np.unique(np.sort(edge_index.T), axis=0).T

    if ret_data:
        # Converts all of these into torch tensors
        X = torch.tensor(X, dtype=torch.float32)
        Y = from_numpy(Y)
        edge_index = to_undirected(from_numpy(edge_index))

        data = Data(x=X, edge_index=edge_index, y=Y)

        train_mask = generate_mask(data_y=data.y, mask=torch.ones(len(data.y),
                                                                  dtype=torch.bool),
                                   num_classes=2, name='Synth',
                                   group_size=train_size // 2, reader=True)

        test_mask = generate_mask(data_y=data.y, mask=~train_mask,
                                  num_classes=2,
                                  name='Synth', group_size=train_size // 2,
                                  reader=True)
        data.test_mask = test_mask
        data.train_mask = train_mask
        data.num_classes = 2
        return data
    print(f'Average distance: {np.mean(distances[:, 1:])}')

    return X, Y, edge_index



# print(f'got a mix of {s} for '
#           f'class_var_scale of {c_var_scale}')
# data = torch.load('Synth_d5')
# X, Y, edge_index = data.x, data.y, data.edge_index

# plot_it(5, class_var_scale=1, X=data.x[:2], Y=data.y,
# edge_index=data.edge_index)

data = generate_data(2000, pos_scale=2, class_var_scale=4, eta=2,
                     near_neighbours=4, train_size=100, map=mapping2,
                     dim=2)
n_points = 300
pos_scale = 1
class_var_scale = 1.18
near_neighbours = 4
X, Y, edge_index = generate_data(n_points, pos_scale=pos_scale,
                                 class_var_scale=class_var_scale,
                                 ret_data=False,
                                 near_neighbours=4, eta=8, map=mapping2)

fname = get_file_name('Synth_d5', False, 'Graph', model_depth=2,
                      prefix='/plots2/')
print(fname)
plot_it(3.5, class_var_scale=class_var_scale, X=X, Y=Y, edge_index=edge_index,
        map=mapping2, eta=8, fname=fname)
print(data)
