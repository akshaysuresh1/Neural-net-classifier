from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from NN_func import *
import os
from astropy.table import Table

#for each network, get the last weights and create a table of the percentages of each
#category is categorized as
#this is going to involve rewriting test() into something that actually works

currentnice = os.nice(0)
desirednice=2
dnice = desirednice-currentnice
currentnice = os.nice(dnice)



valdata = Path("/home/ella2/ebair/data/test_data/generated/")
print("creating validation set")
val_data = create_data(valdata)
print("created validation set")

traindata = Path("/home/ella2/ebair/data/training_data/generated/")
print("creating training set")
train_data = create_data(traindata)
print("created training set")

weightdir = Path("/home/ella2/ebair/networks/5layer4/weights/")
last = 0
for file in os.listdir(weightdir):
     filename = os.fsdecode(file)
     epoch = int(filename.split('_')[-1])
     if epoch > last:
         weightfile = filename
         last = epoch
weight_path = weightdir.joinpath(weightfile)

network = create_network(five_layer_net4, weight_path)
print(table(network, val_data))
print(table(network, train_data))
