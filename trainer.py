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

trainingdata = Path("/home/ella2/ebair/data/training_data/generated/")
testdata = Path("/home/ella2/ebair/data/test_data/generated/")
training_data = create_data(trainingdata)
print("created training set")
test_data = create_data(testdata)
print("created test set")
"""
outputdir = Path("/home/ella2/ebair/networks/3layer/")
network = create_network(three_layer_net)
print("created network")
print("starting training")
trainer(network, training_data, outputdir)
#print("running tests:")
#test(network, test_data)

outputdir = Path("/home/ella2/ebair/networks/4layer/")
network = create_network(four_layer_net)
print("created network")
print("starting training")
trainer(network, training_data, outputdir)
#print("running tests:")
#test(network, test_data)

outputdir = Path("/home/ella2/ebair/networks/5layer/")
network = create_network(five_layer_net)
print("created network")
print("starting training")
trainer(network, training_data, outputdir)
#print("running tests:")
#test(network, test_data)

outputdir = Path("/home/ella2/ebair/networks/5layer2/")
network = create_network(five_layer_net2)
print("created network")
print("starting training")
trainer(network, training_data, outputdir)
#print("running tests:")
#test(network, test_data)
"""
outputdir = Path("/home/ella2/ebair/networks/5layer4/")
network = create_network(five_layer_net4)
print("created network")
print("starting training")
trainer(network, training_data, outputdir)
#print("running tests:")
#test(network, test_data)
