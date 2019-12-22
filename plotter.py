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



directory = Path('/home/ella2/ebair/networks')
folders = ['5layer', '5layer3', '5layer4']
filename = 'accuracy.dat'
valaccuracy = []
trainaccuracy = []
Epoch = []
for folder in folders:
    print(folder)
    V = []
    T = []
    E = []
    path = directory.joinpath(folder)
    path = path.joinpath(filename)
    file = open(path, 'r')
    lines = file.readlines()
    for line in lines:
        print(line)
        line = line.replace('\n', '')
        temp = line.split(' ')
        V.append(float(temp[7]))
        T.append(float(temp[4]))
        E.append(int(temp[1]))
    valaccuracy.append(V)
    trainaccuracy.append(T)
    Epoch.append(E)
lwidth = 2
plt.figure(figsize=(13, 10))
plt.plot(Epoch[0], valaccuracy[0], 'o-', color = 'C0', label="5 Layer Network A Validation Set", linewidth = lwidth)
plt.plot(Epoch[0], trainaccuracy[0], 'o:', color = 'C0', label ='5 Layer Network A Training Set', linewidth = lwidth)
plt.plot(Epoch[1], valaccuracy[1], '^-', color = 'C2',label="5 Layer Network C Validation Set", linewidth = lwidth)
plt.plot(Epoch[1], trainaccuracy[1], '^:', color = 'C2', label="5 Layer Network C Training Set", linewidth = lwidth)
plt.plot(Epoch[2], valaccuracy[2], 's-', color = 'C3', label="5 Layer Network D Validation Set", linewidth = lwidth)
plt.plot(Epoch[2], trainaccuracy[2], 's:', color = 'C3', label = "5 Layer Network D Training Set", linewidth = lwidth)

plt.xlabel('Epochs', fontsize=18)
plt.ylabel("Accuracy %", fontsize=18)
plt.title("Networks Trained to CE Loss of 0.05 on 5 Categories", fontsize=20)
plt.legend()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.savefig(directory.joinpath("accuracy_5layers.png"))
