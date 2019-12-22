from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from NN_func import *
import os

currentnice = os.nice(0)
desirednice=2
dnice = desirednice-currentnice
currentnice = os.nice(dnice)

'''
0 = llbb
1 = llnb
2 = slbb
3 = slnb
4 = noise
'''
exclude = []
outputfolder = 'accuracy.dat'


valdata = Path("/home/ella2/ebair/data/test_data/generated/")
print("creating validation set")
val_data = create_data(valdata, exclude)
print("created validation set")

traindata = Path("/home/ella2/ebair/data/training_data/generated/")
print("creating training set")
train_data = create_data(traindata, exclude)
print("created training set")


directory = Path("/home/ella2/ebair/networks/")
networkfiles = [['5layer4', five_layer_net4]]


j=0
for net in networkfiles:
    print('Testing ' + net[0] + ' network')
    weightdir = directory.joinpath(net[0], "weights")
    output = directory.joinpath(net[0], outputfolder)
    layerval = []
    layertrain = []
    Epoch = []
    for file in os.listdir(weightdir):
         filename = os.fsdecode(file)
         Epoch.append(int(filename.split('_')[-1]))
         path = weightdir.joinpath(filename)
         network = create_network(net[1], path)
         layerval.append(test_simple(network, val_data))
         layertrain.append(test_simple(network, train_data))
    Epoch = np.asarray(Epoch)
    sort = np.argsort(Epoch)
    layerval = np.asarray(layerval)[sort]
    layertrain = np.asarray(layertrain)[sort]
    Epoch = Epoch[sort]
    acfile = open(output, 'w')
    for i in Epoch:
        string = "Epoch: " + str(i) + " Training Accuracy: " + str(layertrain[i-1]) + " Validation Accuracy: " + str(layerval[i-1]) + "\n"
        acfile.write(string)
    acfile.close()
