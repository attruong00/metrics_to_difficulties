# metrics_to_difficulties
train_model.py contains the code to pass difficulty metrics and normalized traversal times through a neural net.
process_data.py contains code to read in data from files and convert it to a PyTorch dataset.

Running in Miniconda environment with Python 3.7, NumPy and PyTorch.

Storing difficulty files in a folder called "diff_files" and path files in a folder called "path_files" outside the metrics_to_difficulties directory.
UPDATE: new metrics files on shared drive, under folder called "norm_metrics_files". These should be used instead of "diff_files"

Currently splitting dataset and giving 250 results to train and the other 50 to test.


**** Changes that need to be pushed (lost connection)
* In data_process.py, I added the line "prev_row, prev_col = row, col" update so that correctly calculates path length.
* In data_process.py, I added resolution = 0.15 at top, and changed line 34 to be "path_length = path_len(np.load(path_file)) * resolution" instead of "path_length = path_len(np.load(path_file))
* In data_process.py, I changed file name to "metrics_x.npy" instead of "difficulties_x.npy"
* In train_model.py line 29, change metrics directory to "norm_metrics_files" instead of "diff_files"
