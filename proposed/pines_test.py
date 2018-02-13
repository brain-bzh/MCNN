'''Test PINES.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse

from utils import progress_bar
import graph
import numpy as np
import pines_aux
import pandas as pd

import numpy as np
import gzip
import pickle


initialized_dataframe = False
class_balancing = False
layers = 2
conv_types = [
        (nn.Conv1d,dict(kernel_size=1)),
        (graph.TernaryLayer, dict(translations_path="translations/translations_1_1_1_radius_small.pkl"))
            ]
dropout = True
epochs = 20
pooling = False
data_augmentation = False
validation = False
count = 0
tests = 10

for conv_type in conv_types:
    for index in range(tests):
        last = pines_aux.train_pines(
            class_balancing=class_balancing,layers=layers,
            conv_type=conv_type,dropout=dropout,epochs=epochs,
            pooling=pooling,data_augmentation=data_augmentation,validation=validation)
        result_dict = dict(type=[conv_type[1]],last_acc=[last],index=[index])
        if not initialized_dataframe:
            dataframe = pd.DataFrame(result_dict,index = [count])
            initialized_dataframe = True
        else:
            dataframe = pd.concat([dataframe,pd.DataFrame(result_dict,index=[count])])
        count += 1
        print(result_dict)


x_train = np.load("../data/PINES/X_train_vol16_1_5.npy")
x_test = np.load("../data/PINES/X_test_vol16_1_5.npy")
y_train = np.load("../data/PINES/y_train_vol16_1_5.npy")
y_test = np.load("../data/PINES/y_test_vol16_1_5.npy")

# In[4]:


from sklearn.neural_network import MLPClassifier
performances = list()
for index in range(tests):
    clf = MLPClassifier(solver='lbfgs', validation_fraction=0)
    clf.fit(x_train, y_train)
    last = clf.score(x_test,y_test)*100
    result_dict = dict(type=["linear"],last_acc=[last],index=[index])
    dataframe = pd.concat([dataframe,pd.DataFrame(result_dict,index=[count])])
    print(result_dict)

    

dataframe.to_csv("tests.csv")
