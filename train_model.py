import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import MetricsDataset

def get_data():
    dataset_dir = '/home/users/atruong/jackal_ws/src/jackal_simulator/jackal_gazebo/worlds/jackal-map-creation/'
    file_prefix = 'difficulties_'

    dataset = []

    num_files = 32
    for i in range(num_files):
        diff_file_name = dataset_dir + file_prefix + str(i) + '.npy'
        diff_file = np.load(diff_file_name)
        dataset.append(diff_file[:-1])
    
    dataset = np.array(dataset)
    print(dataset)
    return dataset

# manipulate inputs to get a test output (mean, variance)
# datapoint is the difficulties array, won't use the average
def fake_result_function(datapoint):
    mean = 0
    
    mean += datapoint[0]
    mean += 3 * datapoint[1]
    mean += datapoint[2] ** 2
    mean += datapoint[3] * 2 + datapoint[4]
    # skip the last metric since it's just the average

    variance = 0
    for i in range(1, len(datapoint)):
        variance += math.fabs(datapoint[i] - datapoint[i - 1])
    
    # add some noise
    mean += random.randint(-1, 1) / 100.0
    variance += random.randint(-1, 1) / 100.0

    return mean, variance


# generate some fake results to test the pipeline
# returns numpy array
def create_fake_results(dataset):
    # dataset obtained from get_data
    num_results = len(dataset)

    result_arr = np.empty([num_results, 2])
    for i in range(num_results):
        result_arr[i][0], result_arr[i][1] = fake_result_function(dataset[i])
    
    return result_arr


class MyDataSet(Dataset):

    # inputs are the diff metrics
    # outputs are the timed results
    def __init__(self, inputs, outputs):
        self.inputs = torch.as_tensor(inputs).float()
        self.outputs = torch.as_tensor(outputs).float()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 4)
        self.fc2 = nn.Linear(4, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # create dataset
    x = get_data()
    y = create_fake_results(x)
    my_data = MyDataSet(x, y)
    # change to Metrics Dataset later
    # Metrics dataset sig: (self, metrics_dir, results_file, path_dir)

    # dataloaders
    # TODO: may not need test_loader (??) if I'm training on the entire batch
    train_loader = DataLoader(my_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(my_data, batch_size=8, shuffle=True)

    net = Net()
    net = net.float()
    lossf = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01)

    # loop over dataset multiple times
    epochs = 1_000
    for i in range(epochs):

        # train model
        # currently not using idx but leaving it there in case
        for idx, (x, y) in enumerate(train_loader):
            pred = net(x)
            loss = lossf(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(loss)

if __name__ == "__main__":
    main()