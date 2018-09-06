import numpy as np
import random


def npy_loader(data_path, label, batch_size):
    data = np.load(data_path)
    labeled_data = list(zip(data, label))
    while True:
        yield random.sample(labeled_data, batch_size)
