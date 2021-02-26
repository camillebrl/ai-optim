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

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader


#############################################################################################
######################### Part1: familiarisation avec Pytorch ###############################
#############################################################################################

tensor = torch.rand(2, 2) # un block de 2 lignes avec 2 éléments par ligne
tensor[:1,:1] = 1 # 1ère ligne 1ère colonne
tensor[1:2,1:2] = 2 # deuxième ligne deuxième colonne
tensor[:1,1:2] = 3 # première ligne 2ème colonne
tensor[1:2,:1] = 4 # deuxième ligne 1ère colonne

tensor.mul(tensor) # met au carré chaque élément de tensor
tensor.matmul(tensor) # multiplie tensor par tensor (multiplication de matrices)
tensor.add_(4) # ajoute 4 à chaque élément de tensor

torch.rand(2,4,4,3) # 2 blocks de 4 dont chacun comporte 4 lignes et 3 éléments dans chaque ligne


model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)




#############################################################################################
#################### Part2: VGG entraîné sur Minicifar avec Pytorch #########################
#############################################################################################


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
        self.classifier = nn.Linear(512, 10) # 

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        # out = torch.nn.functional.softmax(out) # Pas besoin vu qu'on utilise la cross entropy comme loss. Uniquement si on utilisait le min likelihood.
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




def train(epoch,model,optimizer,device,trainloader=trainloader,loss_function=nn.CrossEntropyLoss()):
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


def test(epoch,model,optimizer,device,testloader=testloader,loss_function=nn.CrossEntropyLoss()):
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



def gridsearch(trainloader,validloader,full_trainloader,testloader,loss_function=nn.CrossEntropyLoss(),n_epochs=150):
    learning_rates=[0.001,0.1]
    momentums=[0.9,0.95]
    weight_decay=[5e-5]
    method_gradient_descent=["SGD","ADAM"]
    method_scheduler=["CosineAnnealingLR","ReduceLROnPlateau"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for lr,wd,opt,sch,model_id in product(learning_rates,weight_decay,method_gradient_descent,method_scheduler,cfg.keys()):
        if opt=="SGD":
            for momentum in momentums:
                model=VGG(model_id)
                if device == 'cuda':
                    model = torch.nn.DataParallel(model)
                    cudnn.benchmark = True
                optimizer = optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=wd)
                scheduler=get_schedulers(optimizer,n_epochs)[sch]
                file_name=f"{model_id}_{lr}_{wd}_{opt}_{momentum}_{sch}.csv"
                results,_,best_acc_epoch=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer)
                results.to_csv("./models/"+file_name)

        elif opt=="ADAM":
            model=VGG(model_id)
            if device == 'cuda':
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
            optimizer = optim.Adam(model.parameters(), lr=lr,weight_decay=wd)
            scheduler=get_schedulers(optimizer,n_epochs)[sch]
            file_name=f"{model_id}_{lr}_{wd}_{opt}_None_{sch}.csv"
            results,_,best_acc_epoch=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer)
            results.to_csv("./models/"+file_name)

gridsearch(trainloader,validloader,full_trainloader,testloader)




#############################################################################################
#################### Part3: VGG préentraîné sur Imagenet ####################################
#############################################################################################

class VGG_pretrained(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = models.vgg16(pretrained=True)
        self.classifier = nn.Linear(1000, 10) # VGG de torchvision renvoie 1000 classes

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