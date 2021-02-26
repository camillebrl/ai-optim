### See http://papers.nips.cc/paper/5647-binaryconnect-training-deep-neural-networks-with-b
### for a complete description of the algotihm 


import torch.nn as nn
import numpy
from torch.autograd import Variable
import torch, torchvision
import random

import torch
import torch, torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse

from itertools import product
import pandas as pd
from torchvision import models

import pandas as pd

import numpy as np


def function_separation_on_tensor(nb_bits,tensor):
    nb_bits=np.float16(nb_bits)
    nb=np.float16(1/nb_bits)
    list_nb=[]
    while nb <= 1:
        list_nb.append(np.float16((-1)*nb))
        list_nb.append(np.float16(nb))
        nb+=nb
    list_nb.sort()   
    for i,nb in enumerate(list_nb):
        if nb != -1:
            condition=(tensor<=nb) & (tensor>list_nb[i-1])
        else:
            condition=tensor<=nb
        print(tensor.dtype)
        print(type(nb))
        tensor=torch.where(condition,nb,tensor)
    return tensor




def function_separation_on_tensor_test(nb_bits,tensor):
    nb_bits=np.float16(nb_bits)
    nb=np.float16(1/nb_bits)
    nb_orig=nb
    list_nb=[]
    while nb <= 1.0:
        list_nb.append(np.float16((-1)*nb))
        list_nb.append(np.float16(nb))
        nb+=nb_orig
    list_nb.sort()   
    for i,nb in enumerate(list_nb):
        print(nb)
        if nb == list_nb[-1]:
            condition=tensor>list_nb[i-1]
        elif nb != list_nb[0]:
            condition=(tensor<=nb) & (tensor>list_nb[i-1])
        else:
            condition=tensor<=nb
        t=torch.tensor([nb],device="cuda:0")
        t.half()
        tensor=torch.where(condition,t[0],tensor)
    return tensor




x=torch.randn(2,2,dtype=torch.double,device=torch.device('cuda'))
x=x.half()
print(x)
print(function_separation_on_tensor_test(6,x))