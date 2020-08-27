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
            print(f'Error line')
            break

        val = float(lines[i * 2 + 1])
        if (val == 50.0):
            val = penalty_val
        result[index] = val
    
    return result

def normalize_pathlen(results, paths_dir, num_trials):
    # normalize results by path length
    for i in range(num_worlds):
        path_file = paths_dir + 'path_' + str(i) + '.npy'
            
        path_length = resolution * path_len(np.load(path_file))
        for j in range(num_trials):
            results[j][i] /= path_length

def change_penalty(penalty=50):
    paths_dir = "../test_data/path_files/"

    """
    dwa_file_name = "time_results_10/dwa_results_%d.txt"
    eband_file_name = "time_results_10/eband_results_%d.txt"
    """
    input_file_name = "../test_data/phys_results_avg.txt"
    output_file_name = "../test_data/phys_results.npy"

    """
    # append all timed results to results
    results = []
    
    for i in range(1, 6):
        trial = text_to_array(dwa_file_name % i, num_worlds, penalty)
        results.append(trial)
    
    
    for i in range(1, 6):
        trial = text_to_array(eband_file_name % i, num_worlds, penalty)
        results.append(trial)
    """
    # divide means by length of path
    normalize_pathlen(results, paths_dir, num_trials=5)
    # convert to numpy array
    results = np.asarray(results)

    # changed to do both metrics
    # take the mean of all trials
    means = np.mean(results, axis=0)
    stdevs = np.std(results, axis=0)
    both_stats = [[0 for i in range(2)] for j in range(num_worlds)]
    for i in range(num_worlds):
        both_stats[i][0], both_stats[i][1] = means[i], stdevs[i]

    both_stats = np.asarray(both_stats)
    print(both_stats)
    # save and return
    np.save(output_file_name, both_stats)
    return np.mean(results), np.std(results)
    
# file must exist already
def penalty_stats(penalty=40):
    file_name = "time_results_10/penalty_%d_means.npy" % penalty
    results = np.load(file_name)
    return np.mean(results), np.std(results)

def normalize_pathlen_1d(results, paths_dir, num_envs):
    # normalize results by path length
    for i in range(num_envs):
        path_file = paths_dir + 'path_' + str(i) + '.npy'
        path_length = resolution * path_len(np.load(path_file))
        norm_path_len = results[i] / path_length
        print(i + 1, path_length, results[i], norm_path_len)
        results[i] = norm_path_len

def store_phys_results():
    input_file_name = "../test_data/phys_results_avg.txt"
    output_file_name = "../test_data/phys_results.npy"
    results = text_to_array(input_file_name, num_trials=10, penalty_val=30)
    normalize_pathlen_1d(results, "../test_data/path_files/", 10)
    np.save(output_file_name, results)

def record_pathlens():
    input_dir = '../test_data/path_files/'
    input_file = 'path_%d.npy'
    output_file = '../test_data/path_lengths.txt'

    with open(output_file, 'w') as f:
        for i in range(10):
            path = np.load(input_dir + input_file % i)
            path_length = resolution * path_len(path)
            f.write(f'{i}\n')
            f.write(f'{path_length}\n')


if __name__ == "__main__":
    store_phys_results()