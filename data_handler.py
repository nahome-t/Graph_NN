# Contains functions used to read and write data
import torch
import numpy as np

def generate_mask(data, mask):

    # First
    # Picks 140 examples from the testing data such that each class is
    # picked 20 times, note the testing data is only a subset of the entire
    # data and so the mask for the testing data needs to be given

    # NOTE CLASS SIZE OF 20 SPECIFICALLY GEARED TOWARDS CORA DATASET

    CLASS_SIZE = 20

    data = data.tolist()
    mask = mask.tolist()
    test_mask = [False] * len(data)
    freq = {}
    for i in range(7):
        freq[i] = 0

    for i, x in enumerate(data):
        if mask[i] and freq[x] < CLASS_SIZE:
            freq[x] += 1
            test_mask[i] = True

    return torch.tensor(test_mask)


def count_frequency(filename='tensor'):
    # Gets file, changes it into 2D numpy tensor and effectively counts how
    # often each row or 'function' occurs
    tensor = np.loadtxt(fname=filename, delimiter=',')
    dt = np.dtype((np.void, tensor.dtype.itemsize * tensor.shape[1]))
    b = np.ascontiguousarray(tensor).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(tensor.dtype).reshape(-1, tensor.shape[1])
    print(list(cnt))


def is_consistent(model, data):
    # Returns true if the model perfectly predicts the training data for a
    # dataset which can be filtered from entire dataset using dataset.mask()[]
    mask = data.train_mask
    model.eval()
    with torch.no_grad():
        test_num = mask.sum()
        prediction = model(data).argmax(dim=1)
        correct = (prediction[mask] == data.y[mask]).sum()
        if int(correct) == mask.sum():
            return True
        else:
            return False

    model.train()  # Sets it back to training mode by default