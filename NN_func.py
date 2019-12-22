#from pathlib import Path
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import torch.nn as nn
#import torchvision
#import torchvision.transforms as transforms
import torch.optim as optim
#from torchsummary import summary
import random
import time
import os
import glob
from statistics import mode

# MSE Loss
def MSE_pred_to_tensor(pred):
    t = [[0]*4]
    t[0][pred] = 1
    t = torch.tensor(t)
    return t

# Cross Entropy Loss
def CE_pred_to_tensor(pred):
    t = [pred]
    t = torch.tensor(t)
    return t



# initialize at epoch 1
#notes from Ryan:  create time vs epoch  graph to compare recursion and iteration
#check accuracy as a function of DM, need pulse code
def train(network, training_data, outputdir):
    weights = outputdir.joinpath("weights")
    outputfile = outputdir.joinpath("output.dat")
    # Define a loss function and an optimizer
    # criterion = nn.MSELoss(reduction='sum')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=1e-5)

    # lists to hold loss at each epoch for graphing
    adam_loss = []


    running_loss = len(training_data)
    epoch = 0
    path = outputfile
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    output = open(path, 'w')
    while running_loss > 0.05 * len(training_data):
        running_loss = 0.0
        for i in range(len(training_data)):
            # get the inputs; data is a list of [inputs, labels]
            data_file = training_data[i]
            inputs, labels = data_file[0].to(device), data_file[1].to(device).long()
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = inputs.unsqueeze(0)
            outputs = network(inputs.float())
            # loss = criterion(outputs, labels.float()) # uncomment for MSE
            loss = criterion(outputs, labels)
            # print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print('epoch %d, loss: %.3f' %
            (epoch, running_loss / 10000))
        adam_loss.append(running_loss / 10000)
        string = ("Epoch: " + str(epoch) + "  Running loss: " + str(running_loss / 1000) + "\n")
        output.write(string)
        epoch += 1
        weight_path = weights.joinpath("Epoch_" + str(epoch))
        print("weights saved at " , weight_path)
        if not os.path.exists(weights):
            os.makedirs(weights)
        torch.save(network.state_dict(), weight_path)
        if running_loss <= (0.05 * 10000):
            print ('Finished Training')
            output.close()
            return


# Pair data with labels
def create_data(path, exclude = []):
    data = []
    for plt_cat_snr in path.iterdir():
        plt_cat_snr = np.load(plt_cat_snr, allow_pickle=True)
        plt, category, snr = plt_cat_snr[0], plt_cat_snr[1], plt_cat_snr[2]
        plt, category, snr = torch.tensor(plt).to(device), torch.tensor(category).to(device), torch.tensor(snr).to(device)
        plt = plt.unsqueeze(0) # represents 1 color channel
        if not (category in exclude):
            data.append((plt, category, snr)) # append as tuple (image, label, snr)
    return data # category tensor shape only compatable with Cross Entropy Loss


# Number of correct classifications
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def most_common(lst):
    classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
    if lst == []:
        return 'none'
    else:
        return(classes[mode(lst)])

def avg(lst):
    if len(lst) == 0:
        return -1
    else:
        return sum(lst) / len(lst)

# Define the network
class three_layer_net(nn.Module):

    def __init__(self):
        super(three_layer_net, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=12*122*122, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

# Define the network
class four_layer_net(nn.Module):

    def __init__(self):
        super(four_layer_net, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=24*57*57, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

class five_layer_net(nn.Module):

    def __init__(self):
        super(five_layer_net, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=48*25*25, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

class five_layer_net2(nn.Module):

    def __init__(self):
        super(five_layer_net2, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, stride=1, padding=0),
        nn.ReLU())
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=48*56*56, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

class five_layer_net3(nn.Module):

    def __init__(self):
        super(five_layer_net3, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=6, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=48*26*26, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

class five_layer_net4(nn.Module):

    def __init__(self):
        super(five_layer_net4, self).__init__()

        # input image channels, output channels, NxN square convolutional kernel
        self.layer1 = nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=6, kernel_size=2, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
        nn.Conv2d(in_channels=6, out_channels=12, kernel_size=2, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer3 = nn.Sequential(
        nn.Conv2d(in_channels=12, out_channels=24, kernel_size=7, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        self.layer4 = nn.Sequential(
        nn.Conv2d(in_channels=24, out_channels=48, kernel_size=6, stride=1, padding=0),
        nn.ReLU(), nn.MaxPool2d(kernel_size=4, stride=2))
        # input features, output features
        self.fc1 = nn.Linear(in_features=48*26*26, out_features=5) # linear layer
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # input layer implicit (identity)

        # hidden convolutional layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print(x.shape)

        # flatten matrix
        x = x.reshape(x.size(0), -1)

        # linear layer
        x = self.fc1(x)

        # softmax / output
        # x = self.softmax(x) # don't use softmax with cross-entropy

        return x

def create_network(Network, state_dict=None):
    network = Network().to(device)
    if state_dict != None:
        network.load_state_dict(torch.load(state_dict))
        network.eval()
    return network

def trainer(network, training_data, outputfile):
    # create an instance

    # Train the network
    network = network.float()
    start = time.time()
    print("starting timer")

    train(network, training_data, outputfile)
    end = time.time()
    print("total time = ", end - start) # time in seconds
    return

def table(network, test_data):



    # identify misclassifications

    classes = ['llbb', 'llnb', 'slbb', 'slnb', 'noise']
    distribution = np.zeros((len(classes), len(classes)))
    with torch.no_grad():
        for i in range(len(test_data)):
            data_file = test_data[i]
            inputs, labels, snr = data_file

            inputs = inputs.unsqueeze(0)
            outputs = network(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            distribution[labels.item(), predicted.item()] += 1
    #print(distribution)
    for i in range(distribution.shape[0]):
        norm = np.sum(distribution[i])
        #print(norm)
        distribution[i] = distribution[i]/norm
    #print(distribution[0])

    return (distribution)

#Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test_simple(network, test_data):
    # See how the network performs on the whole dataset
    L = len(test_data)
    correct = 0
    total = 0
    network = network.float()
    with torch.no_grad():
        for i in range(L):
            data_file = test_data[i]
            inputs, labels, _ = data_file

            inputs = inputs.unsqueeze(0)
            outputs = network(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100* correct / total
    print('Accuracy of the network on the %d test images: %d %%' % (L, accuracy))
    return accuracy
