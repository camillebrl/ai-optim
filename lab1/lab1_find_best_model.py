import torch
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

from itertools import product
import pandas as pd
from torchvision import models
import pandas as pd
from os import listdir
from os.path import join
from os.path import isfile
import os
import zipfile
import seaborn as sns
from numpy import *
import math
import matplotlib.pyplot as plt
from pylab import *
import pandas as pd


trainloader= DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader= DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
full_trainloader=DataLoader(minicifar_train,batch_size=32)
testloader=DataLoader(minicifar_test,batch_size=32)

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# Par rapport au modèle VGG de pytorch, une couche a été ajoutée (classfifier) et j'ai ajouté softmax pour avoir % d'appartenance à chaque classe
class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10) # 10 classes

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        out = torch.nn.functional.softmax(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)




def train(epoch,model,optimizer,device,trainloader,loss_function=nn.CrossEntropyLoss()):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc


def test(epoch,model,optimizer,device,testloader,loss_function=nn.CrossEntropyLoss()):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc    

    

def get_schedulers(optimizer,n_epochs):
    schedulers={
        "CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
        "ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min"),
    }
    return schedulers



def train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train(epoch,model,optimizer,device,trainloader,loss_function)  
        valid_acc=test(epoch,model,optimizer,device,validloader,loss_function)
        scheduler.step(valid_acc)
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        dif=train_acc-valid_acc
        if dif > previous_dif:
            overfit_counter+=1
        else:
            overfit_counter=0
        if valid_acc > best_acc_epoch:
            best_acc_epoch=valid_acc
        previous_dif=dif
        epoch+=1
    return pd.DataFrame.from_dict(results).set_index("epoch"),model,best_acc_epoch





device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_function=nn.CrossEntropyLoss()
n_epochs=200

model_path = "./ai-optim-master/lab1/results"
files = [join(model_path,f) for f in listdir(model_path) if isfile(join(model_path,f))] #on ajoute f si f est un fichier

for f in files:
    best_model=False
    df_model=pd.read_csv(f,delimiter=",").set_index("epoch")
    lr=f[f.find("0."):f.find("_5")]
    sch=f[f.rfind("_")+1:f.find(".csv")]
    for acc in df_model["validation_accuracy"]:
        if float(acc) > 85:
            best_model=True
    if best_model==True:
        model_id=f[f.find("VGG"):f.find("_")]
        model=VGG(model_id)
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        if "SGD" in str(f):
            momentum=f[f.find("0.9"):f.rfind("_")]
            optimizer = optim.SGD(model.parameters(), lr=float(lr),momentum=float(momentum), weight_decay=5e-5)
            scheduler=get_schedulers(optimizer,n_epochs)[sch]
            results,_,_=train_model(model,device,loss_function,n_epochs,trainloader,testloader,scheduler,optimizer)
            final_model=model
            results.to_csv("./ai-optim-master/lab1/best_models/"+f[f.find("VGG"):])
        if "ADAM" in str(f):
            optimizer = optim.Adam(model.parameters(), lr=float(lr), weight_decay=5e-5)
            scheduler=get_schedulers(optimizer,n_epochs)[sch]
            results,_,_=train_model(model,device,loss_function,n_epochs,trainloader,testloader,scheduler,optimizer)
            final_model=model
            results.to_csv("./ai-optim-master/lab1/best_models/"+f[f.find("VGG"):])

