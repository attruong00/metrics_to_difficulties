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

        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 1)
        # self.fc4 = nn.Linear(16, 1)
        # self.fc5 = nn.Linear(16, 4)
        # self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # x = F.relu(self.fc5(x))
        x = self.fc3(x)
        return x


def main(penalty=35, alpha=0.001, epochs=10_000):
    # create dataset and split between train and test sets
    metrics_dir = '../norm_metrics_files/'
    results_file = 'time_results_10/penalty_%d_means.npy' % penalty
    my_data = MetricsDataset(metrics_dir, results_file)
    train_data, test_data = torch.utils.data.random_split(my_data, [250, 50])
    print("Train dataset size:", len(train_data))
    print("Test dataset size:", len(test_data))

    # load data
    train_loader = DataLoader(train_data, batch_size=250, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    net = Net()
    lossf = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=alpha)

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
                print('final training loss:', loss)
    
    print('Testing *********************************************')
    with torch.no_grad():

        # test model
        # currently not using idx but leaving it there in case
        running_loss = 0.0
        for idx, (x, y) in enumerate(test_loader):
            pred = net(x)
            pred = torch.squeeze(pred, 1)
            loss = lossf(pred, y)
            running_loss += loss
            # print(loss)
            # print(x, y, pred)
        test_loss = running_loss.item() / 50.0
        print('test loss:', test_loss)
        return test_loss


if __name__ == "__main__":
    main()