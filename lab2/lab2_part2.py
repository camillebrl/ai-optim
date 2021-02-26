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

import datetime as dt
import numpy

trainloader= DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader= DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
full_trainloader=DataLoader(minicifar_train,batch_size=32)
testloader=DataLoader(minicifar_test,batch_size=32)

from minicifar import c10train,c10test


train_cifar10=DataLoader(c10train,batch_size=32)
test_cifar10=DataLoader(c10test,batch_size=32)

class BC():
    def __init__(self, model):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets-1
        self.bin_range = numpy.linspace(start_range,
                end_range, end_range-start_range+1)\
                        .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = [] # This will be used to save the full precision weights
        
        self.target_modules = [] # this will contain the list of modules to be modified

        self.model = model.half() # this contains the model that will be trained and quantified

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range: # capable de binarizer certaines couches et pas d'autres
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)


    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method
        self.save_params()
        ### (2) Binarize the weights in the model, by iterating through the list of target modules and overwrite the values with their binary version 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(torch.sign(self.target_modules[index].data))
            #self.target_modules[index].cpu().detach().apply_(lambda x : -1 if x<0 else 1).cuda() # on ne peut pas appliquer la fonction apply_ avec gpu (uniquement sur cpu)

    def restore(self):

        ### To be completed 
        ### restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):

        ## To be completed 
        ## Clip all parameters to the range [-1,1] using Hard Tanh 
        ## you can use the nn.Hardtanh function
        hth=nn.Hardtanh() # nn.Hardtanh est un Foncteur
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(hth(self.target_modules[index].data))
            #target_modules[index].data.copy_(hth(target_modules[index].detach().data)) # .data permet d'accéder aux données du tensor, et copy_ permet de faire inplace=True
            

    def forward(self,x):

        ### This function is used so that the model can be used while training
        out = self.model(x.half())

        return out


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



cfg_chosen = {}

for i in range(1,9):
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




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel: compte les éléments de p
    # requires_grad: pour déterminer les paramètres que le modèle peut apprendre (car ce sont ceux qui vont jouer dans la descente de gradient)


def train(epoch,bc_model,optimizer,device,trainloader=trainloader,loss_function=nn.CrossEntropyLoss()):
    print('\nEpoch: %d' % epoch)
    bc_model.model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc_model.binarization()
        outputs=bc_model.forward(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        bc_model.restore()
        optimizer.step()
        bc_model.clip()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc


def test(epoch,bc_model,device,testloader):
    bc_model.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bc_model.binarization()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = bc_model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
    acc = 100.*correct/total
    return acc    

    

def get_schedulers(optimizer,n_epochs):
    schedulers={
        "CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
        "ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max"),
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
        valid_acc=test(epoch,model,device,validloader)
        scheduler.step() # il diminue le lr quand la validation accuracy n'augmente plus
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



def models_variant_archi_param(n_epochs=50,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss(),trainloader=trainloader,validloader=validloader,dataset="minicifar"):
    for model_id in cfg_chosen.keys():
        model_brut=VGG(model_id)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            model_brut = torch.nn.DataParallel(model_brut)
            cudnn.benchmark = True
        model=BC(model_brut)
        optimizer = optim.SGD(model.model.parameters(), lr=learning_rate,
                                        momentum=momentum, weight_decay=weight_decay)
        scheduler=get_schedulers(optimizer,n_epochs)[method_scheduler]
        file_name=f"{model_id}_{learning_rate}_{momentum}_{weight_decay}_{method_gradient_descent}_{method_scheduler}.csv"
        results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer)
        results.to_csv(f"./{dataset}/models_binarized_with_dif_nb_param/"+str(count_parameters(model.model))+"_"+file_name)

models_variant_archi_param(trainloader=train_cifar10,validloader=test_cifar10,dataset="cifar10")