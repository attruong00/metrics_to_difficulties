import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset

resolution = 0.15 # meters per pixel
num_worlds = 300

# path is array of points (row, col)
def path_len(path):
    total_dist = 0.0
    prev_row, prev_col = path[0][0], path[0][1]
    for row, col in path[1:]:
        total_dist += math.sqrt((row - prev_row) ** 2 + (col - prev_col) ** 2)
        prev_row, prev_col = row, col
    
    return total_dist

def normalize_results(results_array):

    mean = np.mean(results_array)
    std_dev = np.std(results_array)

    print(mean)
    print(std_dev)

    for i in range(len(results_array)):
        results_array[i] -= mean
        results_array[i] /= std_dev

    return results_array

class MetricsDataset(Dataset):

    # metrics_dir is the folder where the metrics npy arrays are stored
    # results_file is file with all results stored as Nx2 numpy array,
    # each result should have mean traversal time divided by path length
    def __init__(self, metrics_dir, results_file):
        super().__init__()
        self.metrics_dir = metrics_dir

        self.results = np.load(results_file)
        self.n = len(self.results)
        
        # normalize results by mean/std_dev
        # self.results = normalize_results(self.results)

        print('average result', np.mean(self.results))
        print('standard dev result', np.std(self.results))
            
        """
        # quick adjustment to see results without variance
        self.results_no_std = [0 for i in range(self.n)]
        for i in range(self.n):
            self.results_no_std[i] = self.results[i][0]
        """

        # get all difficulty files
        self.diffs = []
        for i in range(self.n):
            diff_file = metrics_dir + 'norm_metrics_' + str(i) + '.npy'

            self.diffs.append(np.load(diff_file)) # 5 metrics

        # convert to torch tensor
        self.diffs = torch.as_tensor(self.diffs).float()
        self.results = torch.as_tensor(self.results).float()
        # self.results_no_std = torch.tensor(self.results_no_std).float()


    def __len__(self):
        return self.n


    def __getitem__(self, index):
        # quick fix to scrap variance
        return self.diffs[index], self.results[index]


def text_to_array(file_name, num_trials, penalty_val):
    # read in lines
    # format: trial number (1-indexed) followed by result on next line
    f = open(file_name, "r")
    lines = f.readlines()
    f.close()

    # create result array
    result = [0 for i in range(num_trials)]

    # store values in result array
    for i in range(num_trials):
        index = int(lines[i * 2])

        # check for repeats or missing results
        if not index == i:
            print("ERROR: expected %d, actual %d" %i %index)
            break

        val = float(lines[i * 2 + 1])
        if (val == 50.0):
            val = penalty_val
        result[index] = val
    
    return result

def normalize_pathlen(results, paths_dir):
    # normalize results by path length
    for i in range(num_worlds):
        path_file = paths_dir + 'path_' + str(i) + '.npy'
            
        path_length = resolution * path_len(np.load(path_file))
        for j in range(10): # 10 trials total
            results[j][i] /= path_length

def change_penalty(penalty=50):
    paths_dir = "../path_files/"
    dwa_file_name = "time_results_10/dwa_results_%d.txt"
    eband_file_name = "time_results_10/eband_results_%d.txt"
    output_file_name = "time_results_10/penalty_%d_means.npy" % penalty

    # append all timed results to results
    results = []
    for i in range(1, 6):
        trial = text_to_array(dwa_file_name % i, num_worlds, penalty)
        results.append(trial)

    for i in range(1, 6):
        trial = text_to_array(eband_file_name % i, num_worlds, penalty)
        results.append(trial)
    
    # divide means by length of path
    normalize_pathlen(results, paths_dir)
    # convert to numpy array
    results = np.asarray(results)
    # take the mean of all trials
    results = np.mean(results, axis=0)
    print(len(results))
    # save and return
    np.save(output_file_name, results)
    return np.mean(results), np.std(results)
    
# file must exist already
def penalty_stats(penalty=40):
    file_name = "time_results_10/penalty_%d_means.npy" % penalty
    results = np.load(file_name)
    return np.mean(results), np.std(results)

if __name__ == "__main__":
    change_penalty()