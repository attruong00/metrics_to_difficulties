import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

# path is array of points (row, col)
def path_len(path):
    total_dist = 0.0
    prev_row, prev_col = path[0][0], path[0][1]
    for row, col in path[1:]:
        total_dist += math.sqrt((row - prev_row) ** 2 + (col - prev_col) ** 2)
    
    return total_dist


class MetricsDataset(Dataset):

    # metrics_dir is the folder where the metrics npy arrays are stored
    # results_file is file with all results stored as Nx2 numpy array,
    # each result should have mean and variance
    # path_dir is the folder with the generated paths
    def __init__(self, metrics_dir, results_file, path_dir):
        super().__init__()
        self.metrics_dir = metrics_dir

        self.results = np.load(results_file)
        self.n = len(self.results)

        # normalize results by path length
        for i in range(self.n):
            path_file = path_dir + 'path_' + str(i) + '.npy'
            path_length = path_len(np.load(path_file))
            self.results[i][0] /= path_length
            self.results[i][1] /= path_length

        # get all difficulty files
        self.diffs = []
        for i in range(self.n):
            diff_file = metrics_dir + 'difficulties_' + str(i) + '.npy'

            self.diffs.append(np.load(diff_file)[:6]) # 6 metrics

        # convert to torch tensor
        self.diffs = torch.as_tensor(self.diffs).float()
        self.results = torch.as_tensor(self.results).float()


    def __len__(self):
        return self.n


    def __getitem__(self, index):
        return self.diffs[index], self.results[index]


def create_datasets(metrics_dir, results_file, path_dir):
    results = np.load(results_file)
    n = len(results)

    # normalize results by path length
    for i in range(n):
        path_file = path_dir + 'path_' + str(i) + '.npy'
        path_length = path_len(np.load(path_file))
        results[i][0] /= path_length
        results[i][1] /= path_length

    # get all difficulty files
    diffs = []
    for i in range(self.n):
        diff_file = metrics_dir + 'difficulties_' + str(i) + '.npy'
        diffs.append(np.load(diff_file)[:6]) # 6 metrics