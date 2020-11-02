# metrics_to_difficulties
train_model.py contains the code to pass difficulty metrics and normalized traversal times through a neural net.
data_process.py contains code to read in data from files and convert it to a PyTorch dataset, as well as several other helper functions for data processing.

Running in Miniconda environment with Python 3.7, NumPy and PyTorch.

Storing difficulty files in a folder called "diff_files" and path files in a folder called "path_files" outside the metrics_to_difficulties directory.
UPDATE: new metrics files on shared drive, under folder called "norm_metrics_files". These should be used instead of "diff_files"

Currently training on all 300 simulation worlds and testing on physical world data. 
The physical world data is included in test_data, so you may need to change the relative path of
the test results because test_data used to be stored in a different folder.
