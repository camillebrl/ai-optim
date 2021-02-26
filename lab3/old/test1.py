import torch
import torch, torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune

import torchvision.transforms as transforms

import os
import argparse

from itertools import product
import pandas as pd
from torchvision import models

import pandas as pd

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader

import numpy as np

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



cfg_chosen = {}

for i in range(2,9):
    model=[]
    for x in cfg["VGG16"]:
        if x != "M":
            model.append(int(x/i))
        else:
            model.append(x)
    cfg_chosen[f"VGG16_{i}"]=model



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg_chosen[vgg_name])
        self.classifier = nn.Linear(cfg_chosen[vgg_name][-2], 10) 
        

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


            

class Pruning():
    def __init__(self,model):
        self.model=model
        self.target_modules = []
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.target_modules.append(m)

    def global_pruning(self,p_to_delete,dim=0):
        for target_module in self.target_modules:
            prune.ln_structured(target_module,name="weight",dim=dim,amount=p_to_delete,n=1) # dim est là où on veut supprimer poids (ligne : 1, col : 0?) Sur quelle dim c'est mieux de pruner?
    def thinet(self,p_to_delete):
        for target_module in self.target_modules:
            #print(target_module.weight.data)
            print()
            print(target_module.weight.data.shape)
            #for channel in range(target_module.weight.data.shape[0]):
                #print(channel)
            # print(target_module.weight.data[0,:,:,:].shape)
            # print(target_module.weight.data[0,:,:,:])
            # print(target_module.inputs)
            # print(target_module.inputs.shape)
            # print(target_module.outputs)
            # print(target_module.outputs.shape)
                #training_set={"w":layer.weight.data,"x":layer.inputs,"y":layer.output}
                #weight_on_which_to_delete_channel=[]
            # weight_to_delete_test=weight_to_delete
            # total_weights=[p.numel() for p in target_module.parameters()]
            # while len(weight_to_delete)<len(total_weights*(1-p_to_delete)):
            #     min_value=np.inf()
            #     for weight in total_weights:
            #         weight_to_delete_test.append(weight)
            #         #Comment je peux comparer l'output du module suivant avec vs sans ce poids?
            #         #Tout ce que je sais comparer c'est self.model et self.model avec le poids enlevé...
            #         value=min(sum(self.model-self.global_pruning(target_module,p_to_delete,dim=3)**2)) #je sais que c'est faux mais je ne vois pas comment faire autrement...
            #         if value>min_value:
            #             min_value=value
            #             min_weight=weight
            #     weight_to_delete.append(min_weight)
                    

        


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel: compte les éléments de p
    # requires_grad: pour déterminer les paramètres que le modèle peut apprendre (car ce sont ceux qui vont jouer dans la descente de gradient)


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


def test(epoch,model,device,testloader=testloader,loss_function=nn.CrossEntropyLoss()):
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



def train_model(model,device,loss_function,n_epochs,trainloader,validloader,full_trainloader,testloader,scheduler,optimizer):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train(epoch,model,optimizer,device,trainloader,loss_function) 
        valid_acc=test(epoch,model,device,validloader,loss_function)
        scheduler.step()
        print(train_acc,valid_acc)
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
    return pd.DataFrame.from_dict(results).set_index("epoch")



def models_variant_archi_param(n_epochs=2,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss()):
    for model_id in cfg_chosen.keys():
        model=VGG(model_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=momentum, weight_decay=weight_decay)
        scheduler=get_schedulers(optimizer,n_epochs)[method_scheduler]
        file_name=f"{model_id}_{learning_rate}_{momentum}_{weight_decay}_{method_gradient_descent}_{method_scheduler}.csv"
        results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,full_trainloader,testloader,scheduler,optimizer)
        #results.to_csv("./models_with_dif_nb_param/"+str(count_parameters(model))+"_"+file_name)
        model_pruning=Pruning(model)
        model_pruning.thinet(0.2)
        #model_pruning.global_pruning(0.1)
        #acc_test=test(n_epochs,model_pruning.model,device)
        #print("test_acc: ",acc_test)
        

models_variant_archi_param()