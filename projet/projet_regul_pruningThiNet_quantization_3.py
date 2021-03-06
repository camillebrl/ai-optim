import torch
import torch, torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import os
import argparse
import datetime as dt
import pandas as pd
from itertools import product
import pandas as pd
from torchvision import models
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
import numpy as np
import torch.nn.utils
import random
import torch.nn.utils.prune as prune

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler

trainloader= DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader= DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(), # prend l'image de manière aléatoire et la tourne en mode miroir
    transforms.ToTensor(), # objigatoire pour pytorch: ne comprend que les tensors
    normalize_scratch, # centre-réduit chaque tensor de l'image
])
# A noter: ici, on ne fait que changer la même image, on ne met pas différentes versions de l'image, ce n'est pas vraiment du data augmentation
# Il aurait fallu prendre le dataset, le multiplier, et appliquer cette transformation (flip, crop) sur la moitié du dataset par exemple

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
]) # On ne fait pas le flip et le crop pour le test


### The data from CIFAR100 will be downloaded in the following dataset
rootdir_cifar100 = './data/cifar100'

c100train = CIFAR100(rootdir_cifar100,train=True,download=True,transform=transform_train)
c100test = CIFAR100(rootdir_cifar100,train=False,download=True,transform=transform_test)

train_cifar100=DataLoader(c100train,batch_size=32)
test_cifar100=DataLoader(c100test,batch_size=32)

### The data from CIFAR10 will be downloaded in the following dataset
rootdir_cifar10 = './data/cifar10'

c10train = CIFAR10(rootdir_cifar10,train=True,download=True,transform=transform_train)
c10test = CIFAR10(rootdir_cifar10,train=False,download=True,transform=transform_test)

train_cifar10=DataLoader(c10train,batch_size=32)
test_cifar10=DataLoader(c10test,batch_size=32)




######################################################################################################################################################
###################################################################VGG################################################################################
######################################################################################################################################################

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


######################################################################################################################################################
###################################################################ResNet#############################################################################
######################################################################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, div=1, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64//div

        self.conv1 = nn.Conv2d(3, 64//div, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64//div)
        self.layer1 = self._make_layer(block, 64//div, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//div, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//div, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//div, num_blocks[3], stride=2)
        self.linear = nn.Linear((512//div)*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            b=block(self.in_planes, planes, stride)
            layers.append(b)
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


nums_blocks={"ResNet18":[2, 2, 2, 2],"ResNet34":[3, 4, 6, 3],"ResNet50":[3, 4, 6, 3],"ResNet101":[3, 4, 23, 3],"ResNet152":[3, 8, 36, 3]}

######################################################################################################################################################
###################################################################ResNext############################################################################
######################################################################################################################################################

class Block(nn.Module):
    '''Grouped convolution block.'''
    expansion = 2

    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        # self.layer4 = self._make_layer(num_blocks[3], 2)
        self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

def ResNeXt29_4x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=4, bottleneck_width=64)

def ResNeXt29_8x64d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=8, bottleneck_width=64)

def ResNeXt29_32x4d():
    return ResNeXt(num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)



######################################################################################################################################################
###################################################################BinaryConnect######################################################################
######################################################################################################################################################

def function_separation_on_tensor(nb_bits,tensor,device):
    array_0=np.linspace(start=-1,stop=1,num=2**nb_bits)
    full_array=np.stack([array_0]*np.prod(tensor.size())).reshape(tensor.size()+(array_0.shape)) # np.stack a une shape de taille a*b*c,4 (si tensor a une taille a*b*c). Nous on veut en sortie a,b,c,4 donc on reshape
    tensor_0=torch.tensor(full_array,device=device)
    x=torch.unsqueeze(tensor,dim=-1) # on transforme tensor en a,b,c,1
    # print(((tensor_0-x)**2).size())
    # print((torch.argmin((tensor_0-x)**2,dim=-1)))
    results=torch.gather(tensor_0, dim=-1,index=torch.unsqueeze(torch.argmin((tensor_0-x)**2,dim=-1),dim=-1))
    return results.view(tensor.size()) # gather a besoin d'avoir les mêmes dimensions pour les 2
    #return tensor_0[torch.argmin((tensor_0-x)**2,dim=-1)] # on change le tensor en mettant dedans les valeurs les plus proches des valeurs à la même place de tensor_0


class BC():
    def __init__(self,model,nb_bits,device):

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

        self.device=device

        self.nb_bits=nb_bits

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
            print(index)
            print(self.target_modules[index].data.size())
            print(function_separation_on_tensor(self.nb_bits,self.target_modules[index].data,self.device).size())
            self.target_modules[index].data.copy_(function_separation_on_tensor(self.nb_bits,self.target_modules[index].data,self.device))
            #self.target_modules[index].cpu().detach().apply_(lambda x : -1 if x<0 else 1).cuda() # on ne peut pas appliquer la fonction apply_ avec gpu (uniquement sur cpu)

    def restore(self):

        ### restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):

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

######################################################################################################################################################
###################################################################Regularisation#####################################################################
######################################################################################################################################################

class Orthogo():
    def __init__(self,model,device):
        self.model=model
        self.device=device
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.target_modules.append(m)
    def soft_orthogonality_regularization(self,reg_coef):
        regul=0.
        for i,m in enumerate(self.target_modules):
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            reg_coef_i=reg_coef
            regul+=reg_coef_i*torch.norm(torch.transpose(w,0,1).matmul(w)-torch.eye(height,device=self.device))**2
            #reg_grad=4*reg_coef*w*(torch.transpose(w,0,1)*w-torch.eye(height))
        return regul # le terme de régularisation est sur tous les modules! (somme)

    def double_soft_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.norm(torch.transpose(w,0,1).matmul(w)-torch.eye(height,device=self.device))**2 + torch.norm(w.matmul(torch.transpose(w,0,1))-torch.eye(height,device=self.device))**2)
        return regul
    def mutual_coherence_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.max(torch.abs((torch.transpose(w,0,1).matmul(w)-torch.eye(height,device=self.device))**2)))
        return regul
    def spectral_isometry_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            x=random.random()
            u=w.dot(x)
            v=w.dot(u)
            regul+=reg_coef*(torch.sum(v**2,dim=-1)/torch.sum(u**2,dim=-1))
        return regul

######################################################################################################################################################
###################################################################pruning############################################################################
######################################################################################################################################################

def hook(module, input, output):
    if hasattr(module,"_input_hook"):
        module._input_hook=input[0] # input contient plusieurs inputs différents? ça ne fonctionne sans [0] car ça renvoie un tuple...
        module._output_hook=output
    else:
        setattr(module,"_input_hook",input[0])
        setattr(module,"_output_hook",output)

class Pruning():
    def __init__(self,model,device):
        self.model=model
        self.device=device
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m,Bottleneck):
                convs=[]
                for m_2 in m.modules():
                    if isinstance(m_2,nn.Conv2d):
                        convs.append(m_2)
                self.target_modules.extend(convs[:-1]) # on prend toutes les conv2d de chaque block sauf la dernière
        print(self.target_modules)
    def thinet(self,p_to_delete):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(hook) # register_forward_hook prend un objet fonction en paramètre
        for mod,m in enumerate(self.target_modules): # Dans le papier, il est indiqué que 90% des floating points operattions sont contenus dans les 10 premiers layers
            print("module:",mod)
            if isinstance(m, nn.Conv2d):    
                list_training=[]
                n=64
                subset_indices = [random.randint(0,len(trainloader)-1) for _ in range(n)] # récupère au hasard les indices de n batch dans trainloader
                for i, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if i in subset_indices:
                        for j in range(inputs.size()[0]):
                            size=inputs.size()
                            self.model(inputs[j].view((1,size[1],size[2],size[3])))
                            channel=random.randint(0,m._output_hook.size()[1]-1)
                            ligne=random.randint(0,m._output_hook.size()[2]-1)
                            colonne=random.randint(0,m._output_hook.size()[3]-1)
                            w=m.weight.data[channel,:,:,:] # W = output_channel * input_channel * ligne * colonne
                            #np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                            #x_2=torch.pad(m._input_hook[i][j,:,:,:],((0,0),(1,1),(1,1))) # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                            x_2=torch.zeros((m._input_hook[0].size()[0],m._input_hook[0].size()[1]+2,m._input_hook[0].size()[2]+2),device=self.device)
                            x_2[:,1:-1,1:-1] = m._input_hook[0] # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                            x=x_2[:,ligne:ligne+w.size()[1],colonne:colonne+w.size()[2]] # On ne prend pas -1 car le décalage est déjà là de base
                            list_training.append(x*w)
                            
                channels_to_delete=[]
                channels_to_try_to_delete=[]
                total_channels=[i for i in range(m._input_hook.size()[1])]
                
                c=len(total_channels)
                
                while len(channels_to_delete)<c*p_to_delete:
                    
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
                    
                m.weight.data[:,channels_to_delete,:,:]=torch.zeros(m.weight.data[:,channels_to_delete,:,:].size(),device=self.device)
                #Pour simplifier, on ne supprime pas vraiment les poids à enlever mais on les met à 0, car si on devait les supprimer, il faudrait supprimer les channels en input aussi, et faire une sorte de "backpropagation", ce qui est trop compliqué et je n'ai pas le temps
                #m.weight.data=m.weight.data[:,total_channels,:,:] # Car total_channels ne contient que les poids que l'on garde
                #m._input_hook[i]=m._input_hook[i][total_channels,:,:]


######################################################################################################################################################
###################################################################Application_model##################################################################
######################################################################################################################################################

def train(epoch,model,optimizer,device,trainloader,loss_function,model_orthogo,function,reg_coef):
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
        if function == "simple":
            loss+=model_orthogo.soft_orthogonality_regularization(reg_coef)
        elif function == "double":
            loss+=model_orthogo.double_soft_orthogonality_regularization(reg_coef)
        elif function == "mutual_coherence":
            loss+=model_orthogo.mutual_coherence_orthogonality_regularization(reg_coef)
        elif function == "spectral_isometry":
            loss+=model_orthogo.spectral_isometry_orthogonality_regularization(reg_coef)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc



def train_quantization(epoch,bc_model,optimizer,device,trainloader,loss_function):
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



def validation(epoch,model,device,validloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc    



def validation_quantization(epoch,bc_model,device,validloader):
    bc_model.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bc_model.binarization()
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = bc_model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
    acc = 100.*correct/total
    return acc    


def validation_half(epoch,model,device,validloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.half())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc    


def get_schedulers(optimizer,n_epochs):
    schedulers={
        "CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
        "ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max"),
    }
    return schedulers




def train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo,function,reg_coef):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train(epoch,model,optimizer,device,trainloader,loss_function,model_orthogo,function,reg_coef) 
        valid_acc=validation(epoch,model,device,validloader)
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



def train_model_quantization(bc_model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train_quantization(epoch,bc_model,optimizer,device,trainloader,loss_function) 
        valid_acc=validation_quantization(epoch,bc_model,device,validloader)
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



reg_coefs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
functions=["simple","spectral_isometry"]
pruning_rate=[0.2,0.3,0.4,0.5,0.6]
nb_bits_list=[i for i in range(2,10)]

def models_variant_archi_param(trainloader,validloader,n_epochs=150,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss(),dataset="minicifar"):
    for model_name,model_nb_blocks in zip(nums_blocks.keys(),nums_blocks.values()):
        for div_param in range(1,9):
            model=ResNet(Bottleneck,model_nb_blocks,div=div_param,num_classes=int(dataset[dataset.find("1"):]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
            #model.to(torch.device("cuda:0"))
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=momentum, weight_decay=weight_decay)
            scheduler=get_schedulers(optimizer,n_epochs)[method_scheduler]
            for function in functions:
                for reg_coef in reg_coefs:
                    model_orthogo=Orthogo(model,device)
                    results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo,function,reg_coef)
                    file_name=f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv"
                    for rate in pruning_rate:
                        model_pruning=Pruning(model,device)
                        model_pruning.thinet(rate)
                        file_name=f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv"
                        results_pruning={"accuracy":[]}
                        results_pruning["accuracy"].append(validation(n_epochs,model_pruning.model,device,validloader))
                        results_pruning_df=pd.DataFrame.from_dict(results_pruning)
                        torch.save(model_pruning.model.state_dict(),f"./{dataset}/model_{function}_reg_dif_para/models/"+f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                        results_pruning_df.to_csv(f"./{dataset}/model_{function}_reg_dif_para/results/"+f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                        results_pruning_retrained=train_model(model_pruning.model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,None,None,None)
                        torch.save(model_pruning.model.state_dict(),f"./{dataset}/model_{function}_reg_dif_para/models/"+f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                        results_pruning_retrained.to_csv(f"./{dataset}/model_{function}_reg_dif_para/results/"+f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                        model_pruned_half=model_pruning.model
                        model_pruned_half.half()
                        results_half_precision={"accuracy":[]}
                        results_half_precision["accuracy"].append(validation_half(n_epochs,model_pruned_half,device,validloader))
                        results_half_precision_df=pd.DataFrame.from_dict(results_half_precision)
                        torch.save(model_pruned_half.state_dict(),f"./{dataset}/model_{function}_reg_dif_para/models/"+f"half_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                        results_half_precision_df.to_csv(f"./{dataset}/model_{function}_reg_dif_para/results/"+f"half_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                        for nb_bits in nb_bits_list:
                            bc_model=BC(model_pruning.model,nb_bits,device)
                            results_bc_model=train_model_quantization(bc_model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer)
                            torch.save(bc_model.model.state_dict(),f"./{dataset}/model_{function}_reg_dif_para/models/"+f"half_quantized_with_BinaryConnect_precisionof_{nb_bits}bits_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                            results_bc_model.to_csv(f"./{dataset}/model_{function}_reg_dif_para/results/"+f"half_quantized_with_BinaryConnect_precisionof_{nb_bits}bits_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
# Est-ce que j'ai besoin, à chaque fois que je modifie le modèle, de réinitialiser l'optimizer & le scheduler?


for dataset in ["cifar10","cifar100"]:
    if dataset == "cifar10":
        trainloader=train_cifar10
        testloader=test_cifar10
    elif dataset == "cifar100":
        trainloader=train_cifar100
        testloader=test_cifar100
    models_variant_archi_param(trainloader=trainloader,validloader=testloader,dataset=dataset)