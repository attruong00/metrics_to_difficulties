import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import MetricsDataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(5, 1)
        # self.fc2 = nn.Linear(64, 1)
        # self.fc3 = nn.Linear(16, 1)
        # self.fc4 = nn.Linear(16, 1)
        # self.fc5 = nn.Linear(16, 4)
        # self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc1(x)
        return x


def main(penalty=30, alpha=0.01, epochs=1_000):
    # create dataset and split between train and test sets
    metrics_dir = '../norm_metrics_files/'
    results_file = 'time_results_10/penalty_%d_means.npy' % penalty
    test_metrics_dir = '../test_data/norm_metrics_files/'
    test_results_file = '../test_data/phys_results.npy'

    """
    my_data = MetricsDataset(metrics_dir, results_file)
    train_data, test_data = torch.utils.data.random_split(my_data, [250, 50])
    
    """
    train_data = MetricsDataset(metrics_dir, results_file)
    test_data = MetricsDataset(test_metrics_dir, test_results_file)
    

    # load data
    train_loader = DataLoader(train_data, batch_size=300, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    net = Net()
    lossf = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=alpha)

    # print params
    print('*****************************')
    print('Train dataset size:', len(train_data))
    print('Test dataset size:', len(test_data))
    print('Layers', 1)
    print('Penalty', penalty)
    print('Learning rate', alpha)
    print('Epochs', epochs)
    print()
    print('Training...')
    # loop over dataset multiple times
    # epochs = 10_000
    for i in range(epochs):

        # train model
        # currently not using idx but leaving it there in case
        for idx, (x, y) in enumerate(train_loader):
            pred = net(x)
            pred = torch.squeeze(pred, 1)
            loss = lossf(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i == epochs - 1:
                print(f'final training loss {i}:', loss.item())
    
    print('Testing...')
    with torch.no_grad():

        # test model
        running_loss = 0.0
        predictions = []
        for idx, (x, y) in enumerate(test_loader):
            pred = net(x)
            pred = torch.squeeze(pred, 1)
            loss = lossf(pred, y)

            # keep track of predictions and loss
            running_loss += loss
            predictions.append(pred.item())
            # print(loss)
            print(idx, x, y.item(), pred.item())

        test_loss = running_loss.item() / 50.0
        print('test loss:', test_loss)
        return predictions


if __name__ == "__main__":
    main()