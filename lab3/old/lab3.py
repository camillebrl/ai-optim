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
import random

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
        # for i,layer in enumerate(self.features):
        #     x{i}=layer(x)
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




            

def hook(module, input, output):
    if hasattr(module,"_inputs_hook"):
        module._inputs_hook.append(input[0]) # input contient plusieurs inputs différents? ça ne fonctionne sans [0] ; on n'a qu'un seul batch dans input
    else:
        setattr(module,"_inputs_hook",[input[0]])
    # if hasattr(module,"_outputs_hook"):
    #     module._outputs_hook.append(output)
    # else:
    #     setattr(module,"_outputs_hook",[output])





class Pruning():
    def __init__(self,model):
        self.model=model
        self.target_modules = []
        self.saved_params = [] # This will be used to save the full precision weights

    def save_params(self):
        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters
        for index in range(len(self.target_modules)):
            self.saved_params[index].copy_(self.target_modules[index].data)
    
    def restore(self):
        ### restore the copy from self.saved_params into the model 
        for index in range(len(self.target_modules)):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def global_pruning(self,p_to_delete,dim=0):
        for target_module in self.target_modules:
            prune.ln_structured(target_module,name="weight",dim=dim,amount=p_to_delete,n=1) # dim est là où on veut supprimer poids (ligne : 1, col : 0?) Sur quelle dim c'est mieux de pruner?
    
    def thinet(self,p_to_delete):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(hook) # register_forward_hook prend un objet fonction en paramètre
        test(0,self.model,device="cuda:0", testloader=trainloader) # on doit faire un test car sinon la fonction register_forward_hook n'a pas accès au forward
        for i,m in enumerate(self.target_modules[:7]): # Dans le papier, il est indiqué que 90% des floating points operattions sont contenus dans les 10 premiers layers
            print(i)
            if isinstance(m, nn.Conv2d):    
                list_training=[]
                for i in range(len(m._inputs_hook[:2])): # on itère sur le nombre de batchs (il y a 100 batchs ici, on a bien len(m._inputs_hook)=100)
                    for j in range(len(m._inputs_hook[i])):
                        output=m(m._inputs_hook[i])
                        channel=random.randint(0,output.size()[1]-1)
                        ligne=random.randint(0,output.size()[2]-1)
                        colonne=random.randint(0,output.size()[3]-1)
                        w=m.weight.data[channel,:,:,:] # W = output_channel * input_channel * ligne * colonne
                        #np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                        #x_2=torch.pad(m._inputs_hook[i][j,:,:,:],((0,0),(1,1),(1,1))) # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                        x_2=torch.zeros((m._inputs_hook[i][j].size()[0],m._inputs_hook[i][j].size()[1]+2,m._inputs_hook[i][j].size()[2]+2),device="cuda:0")
                        x_2[:,1:-1,1:-1] = m._inputs_hook[i][j] # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                        x=x_2[:,ligne:ligne+w.size()[1],colonne:colonne+w.size()[2]] # On ne prend pas -1 car le décalage est déjà là de base
                        list_training.append(x*w)
                channels_to_delete=[]
                channels_to_try_to_delete=[]
                total_channels=[i for i in range(m._inputs_hook[0].size()[1])]
                c=len(total_channels)
                while len(channels_to_delete)<c*(1-p_to_delete):
                    min_value=np.inf
                    for channel in total_channels:
                        channels_to_try_to_delete=channels_to_delete+[channel]
                        value=0
                        for a in list_training:
                            a_changed=a[channels_to_try_to_delete,:,:]
                            result=torch.sum(a_changed)
                            value+=result**2
                        if value<min_value:
                            min_value=value
                            min_channel=channel
                    channels_to_delete.append(min_channel)
                    total_channels.remove(min_channel)
                print(len(total_channels))
                m.weight.data=m.weight.data[:,total_channels,:,:] # Car total_channels ne contient que les poids que l'on garde

                
    def connectionpruning(self,p_to_delete):
        # On prune les connections, et après les avoir prunées, on prune les neurones avec 0 output connections et 0 input connections.
        # Le nombre de connection au layer i c'est Ci=nombre de neurones au layer i * nombre de neurones au layer i-1
        # On a aussi une formule avec le dropout (chaque paramètre est droppé avec une certaine probabilité pendant le training, mais ces paramètres reviennent pendantl l'interférence. Contrairement au pruning où les paramètres sont droppé forever.)
        # dropout_rate_pdt_reapprentissage=original_dropout_rate * racine (nombre de connections après le réapprentissage / nombrede connections à l'origine).
        pass
    def filterpruning(self,p_to_delete):
        # For each filterFi,j, calculate the sum of its absolute kernel weightssj=∑nil=1∑|Kl|
        # Sort the filters bysj
        # Prunemfilters with the smallest sum values and their corresponding feature maps. Thekernels in the next convolutional layer corresponding to the pruned feature maps are alsoremoved
        # A new kernel matrix is created for both theith andi+ 1th layers, and the remaining kernelweights are copied to the new model
        pass

        


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

rate_to_delete=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def models_variant_archi_param(n_epochs=10,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss()):
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
        nb_parameter_in_vgg=count_parameters(model)
        #results.to_csv("./models_with_dif_nb_param/"+str(count_parameters(model))+"_"+file_name)
        model_pruning=Pruning(model)
        #model_pruning.save_params()
        dict_thinet={"accuracy":[],"rate_of_channels_deleted":[]}
        for rate in rate_to_delete:
            model_pruning.thinet(rate)
            acc_test=test(n_epochs,model_pruning.model,device)
            print("rate:",rate,"acc_test",acc_test)
            dict_thinet["accuracy"].append(acc_test)
            dict_thinet["rate_of_channels_deleted"].append(rate)
            model_pruning.restore()
        pd.DataFrame.from_dict(dict_thinet).to_csv("./thinet_with_dif_nb_param/"+str(nb_parameter_in_vgg)+"thinet.csv")
        #model_pruning.global_pruning(0.1)
        #acc_test=test(n_epochs,model_pruning.model,device)
        #print("test_acc: ",acc_test)
        

models_variant_archi_param()