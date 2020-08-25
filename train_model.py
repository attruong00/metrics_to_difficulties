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

        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, 16)
        self.fc5 = nn.Linear(16, 4)
        self.fc6 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


def main():
    # create dataset and split between train and test sets
    my_data = MetricsDataset('../norm_metrics_files/', '../nn_input_1.npy', '../path_files/')
    train_data, test_data = torch.utils.data.random_split(my_data, [250, 50])
    print("Train dataset size:", len(train_data))
    print("Test dataset size:", len(test_data))

    # load data
    train_loader = DataLoader(train_data, batch_size=250, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    net = Net()
    lossf = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)

    # loop over dataset multiple times
    epochs = 100_000
    for i in range(epochs):

        # train model
        # currently not using idx but leaving it there in case
        for idx, (x, y) in enumerate(train_loader):
            pred = net(x)
            loss = lossf(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epochs % 100 == 0:
                print(loss)
    
    print('Testing *********************************************')
    with torch.no_grad():

        # test model
        # currently not using idx but leaving it there in case
        for idx, (x, y) in enumerate(test_loader):
            pred = net(x)
            loss = lossf(pred, y)
            # print(loss)
            print(pred, y)


if __name__ == "__main__":
    main()