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
        self.bin_range = np.linspace(start_range,
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
    if hasattr(module,"_input_hook"):
        module._input_hook=input[0] # input contient plusieurs inputs différents? ça ne fonctionne sans [0] car ça renvoie un tuple...
        module._output_hook=output
    else:
        setattr(module,"_input_hook",input[0])
        setattr(module,"_output_hook",output)






class Pruning():
    def __init__(self,model):
        self.bc_class=model
        self.model=model.model
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.target_modules.append(m)

    def global_pruning(self,p_to_delete,dim=0):
        for target_module in self.target_modules:
            prune.ln_structured(target_module,name="weight",dim=dim,amount=p_to_delete,n=1) # dim est là où on veut supprimer poids (ligne : 1, col : 0?) Sur quelle dim c'est mieux de pruner?
    
    def thinet(self,p_to_delete,nb_of_layer_to_prune=10):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(hook) # register_forward_hook prend un objet fonction en paramètre
        torch.cuda.empty_cache()
        for mod,m in enumerate(self.target_modules[:nb_of_layer_to_prune]): # Dans le papier, il est indiqué que 90% des floating points operattions sont contenus dans les 10 premiers layers
            print("module:",mod)
            if isinstance(m, nn.Conv2d):    
                list_training=[]
                n=64
                subset_indices = [random.randint(0,len(trainloader)-1) for _ in range(n)] # récupère au hasard les indices de n batch dans trainloader
                for i, (inputs, targets) in enumerate(trainloader):
                    if i in subset_indices:
                        for j in range(inputs.size()[0]):
                            size=inputs.size()
                            self.model(inputs[j].view((1,size[1],size[2],size[3])).half())
                            channel=random.randint(0,m._output_hook.size()[1]-1)
                            ligne=random.randint(0,m._output_hook.size()[2]-1)
                            colonne=random.randint(0,m._output_hook.size()[3]-1)
                            w=m.weight.data[channel,:,:,:] # W = output_channel * input_channel * ligne * colonne
                            torch.cuda.empty_cache()
                            #np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                            #x_2=torch.pad(m._input_hook[i][j,:,:,:],((0,0),(1,1),(1,1))) # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                            x_2=torch.zeros((m._input_hook[0].size()[0],m._input_hook[0].size()[1]+2,m._input_hook[0].size()[2]+2),device="cuda:0")
                            torch.cuda.empty_cache()
                            x_2[:,1:-1,1:-1] = m._input_hook[0] # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                            torch.cuda.empty_cache()
                            x=x_2[:,ligne:ligne+w.size()[1],colonne:colonne+w.size()[2]] # On ne prend pas -1 car le décalage est déjà là de base
                            torch.cuda.empty_cache()
                            list_training.append(x*w)
                            torch.cuda.empty_cache()
                channels_to_delete=[]
                channels_to_try_to_delete=[]
                total_channels=[i for i in range(m._input_hook.size()[1])]
                torch.cuda.empty_cache()
                c=len(total_channels)
                torch.cuda.empty_cache()
                while len(channels_to_delete)<c*p_to_delete:
                    torch.cuda.empty_cache()
                    min_value=np.inf
                    for channel in total_channels:
                        channels_to_try_to_delete=channels_to_delete+[channel]
                        torch.cuda.empty_cache()
                        value=0
                        for a in list_training:
                            a_changed=a[channels_to_try_to_delete,:,:]
                            torch.cuda.empty_cache()
                            result=torch.sum(a_changed)
                            torch.cuda.empty_cache()
                            value+=result**2
                            torch.cuda.empty_cache()
                        if value<min_value:
                            min_value=value
                            min_channel=channel
                    channels_to_delete.append(min_channel)
                    torch.cuda.empty_cache()
                    total_channels.remove(min_channel)
                    torch.cuda.empty_cache()
                m.weight.data[:,channels_to_delete,:,:]=torch.zeros(m.weight.data[:,channels_to_delete,:,:].size(),device="cuda:0").half()
                #Pour simplifier, on ne supprime pas vraiment les poids à enlever mais on les met à 0, car si on devait les supprimer, il faudrait supprimer les channels en input aussi, et faire une sorte de "backpropagation", ce qui est trop compliqué et je n'ai pas le temps
                #m.weight.data=m.weight.data[:,total_channels,:,:] # Car total_channels ne contient que les poids que l'on garde
                #m._input_hook[i]=m._input_hook[i][total_channels,:,:]

                
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
    return sum(p.numel() for p in model.model.parameters() if p.requires_grad)
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
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        bc_model.binarization()
        torch.cuda.empty_cache()
        outputs=bc_model.forward(inputs)
        torch.cuda.empty_cache()
        loss = loss_function(outputs, targets)
        loss.backward()
        torch.cuda.empty_cache()
        bc_model.restore()
        torch.cuda.empty_cache()
        optimizer.step()
        torch.cuda.empty_cache()
        bc_model.clip()
        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            torch.cuda.empty_cache()
            inputs, targets = inputs.to(device), targets.to(device)
            torch.cuda.empty_cache()
            outputs = bc_model.forward(inputs)
            torch.cuda.empty_cache()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
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


rate_to_delete=[0.2]
#rate_to_delete=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

def models_variant_archi_param(n_epochs=2,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss()):
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
        count_parameter=count_parameters(model)
        results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer)
        resultat={"accuracy":[],"rate":[]}
        for rate in rate_to_delete:
            model_pruning=Pruning(model)
            model_pruning.thinet(rate)
            torch.cuda.empty_cache()
            resultat["accuracy"].append(test(n_epochs,model_pruning.bc_class,device,validloader))
            resultat["rate"].append(rate)
        #pd.DataFrame.from_dict(resultat).to_csv("./binarized_thinet_with_dif_nb_param/"+"thinet"+str(count_parameter)+file_name)
        #model_pruning.global_pruning(0.1)
        #acc_test=test(n_epochs,model_pruning.model,device)
        #print("test_acc: ",acc_test)
        

models_variant_archi_param()