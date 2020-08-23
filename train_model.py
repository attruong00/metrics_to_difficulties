import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from data_process import MetricsDataset



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

        self.fc1 = nn.Linear(6, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc3 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    # create dataset and split between train and test sets
    my_data = MetricsDataset('../diff_files/', '../nn_input_dwa_only.npy', '../path_files/')
    train_data, test_data = torch.utils.data.random_split(my_data, [250, 50])
    print("Train dataset size:", len(train_data))
    print("Test dataset size:", len(test_data))

    # load data
    train_loader = DataLoader(train_data, batch_size=250, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    net = Net()
    lossf = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.003)

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
    
    print('Testing *********************************************')
    with torch.no_grad():

        # test model
        # currently not using idx but leaving it there in case
        for idx, (x, y) in enumerate(test_loader):
            pred = net(x)
            # loss = lossf(pred, y)
            # print(loss)
            print(pred, y)


if __name__ == "__main__":
    main()