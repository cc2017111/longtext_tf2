import numpy as np


def data_shuffle(dataset):
    sh_index = np.arange(len(dataset))
    np.random.shuffle(sh_index)
    shuffle_dataset = []
    for i in range(len(dataset)):
        shuffle_dataset.append(dataset[sh_index[i]])
    return shuffle_dataset
