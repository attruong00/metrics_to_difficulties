import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

resolution = 0.15 # meters per pixel

# path is array of points (row, col)
def path_len(path):
    total_dist = 0.0
    prev_row, prev_col = path[0][0], path[0][1]
    for row, col in path[1:]:
        total_dist += math.sqrt((row - prev_row) ** 2 + (col - prev_col) ** 2)
        prev_row, prev_col = row, col
    
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
            
            path_length = resolution * path_len(np.load(path_file))
            self.results[i][0] /= path_length
            self.results[i][1] /= path_length
            
        # quick adjustment to see results without variance
        self.results_no_std = [0 for i in range(self.n)]
        for i in range(self.n):
            self.results_no_std[i] = self.results[i][0]

        # get all difficulty files
        self.diffs = []
        for i in range(self.n):
            diff_file = metrics_dir + 'norm_metrics_' + str(i) + '.npy'

            self.diffs.append(np.load(diff_file)) # 5 metrics

        # convert to torch tensor
        self.diffs = torch.as_tensor(self.diffs).float()
        self.results = torch.as_tensor(self.results).float()
        self.results_no_std = torch.tensor(self.results_no_std).float()


    def __len__(self):
        return self.n


    def __getitem__(self, index):
        # quick fix to scrap variance
        return self.diffs[index], self.results_no_std[index]


def text_to_array(file_name, num_trials, special_val):
    f = open(file_name, "r")

    lines = f.readlines()

    result = [0 for i in range(num_trials)]

    for i in range(num_trials):
        index = int(lines[i * 2])

        if not index == i:
            print("ERROR: expected %d, actual %d" %i %index)
            break

        val = float(lines[i * 2 + 1])
        if (val == 50.0):
            val = special_val
        result[index] = val
    
    return result

def main():
    dwa_file_name = "timing_results/dwa_results_%d.txt"

    results = []
    for i in range(1, 4):
        trial = text_to_array(dwa_file_name % i, 300, 40.0)
        results.append(trial)
        
        print(i)
        print(trial)

if __name__ == "__main__":
    main()