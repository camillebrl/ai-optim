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
from torchvision.datasets import CIFAR100
import numpy as np

rootdir = './data/cifar100'
trainloader = CIFAR100(rootdir,train=True,download=True)
testloader = CIFAR100(rootdir,train=False,download=True)



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
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


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
###################################################################BinaryConnect############################################################################
######################################################################################################################################################

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



class Orthogo():
    def __init__(self,model):
        self.model=model
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.target_modules.append(m)
    def soft_orthogonality_regularization(self,reg_coef):
        regul=0.
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.transpose(w,0,1).dot(w)-torch.eye(height))**2
            #reg_grad=4*reg_coef*w*(torch.transpose(w,0,1)*w-torch.eye(height))
        return regul # le terme de régularisation est sur tous les modules! (somme)
            # on peut même tester avec plusieurs reg_coefs (en fonction des modules)
            # bien régulariser les 1ères couches est interressant, et moins bien celles d'après. Donc le reg_coef doit être plus grand
            # plus reg_coef est grand, plus il sera régularisé. On peut faire un regul_coef variable.
            # On peut tester de diviser reg_coef d'un module à l'autre (voir comme les méthodos du scheduler)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel: compte les éléments de p
    # requires_grad: pour déterminer les paramètres que le modèle peut apprendre (car ce sont ceux qui vont jouer dans la descente de gradient)


def train(epoch,model,optimizer,device,model_orthogo,trainloader=trainloader,loss_function=nn.CrossEntropyLoss()):
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
        #loss=loss+model_orthogo.soft_orthogonality_regularization(0.2)
        loss.backward()
        optimizer.step()
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


    

def get_schedulers(optimizer,n_epochs):
    schedulers={
        "CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
        "ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max"),
    }
    return schedulers




def train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train(epoch,model,optimizer,device,model_orthogo,trainloader,loss_function) 
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




def models_variant_archi_param(n_epochs=2,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss(),trainloader=trainloader,validloader=testloader,dataset="minicifar"):
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
        #results.to_csv(f"./{dataset}/models_binarized_with_dif_nb_param/"+str(count_parameters(model.model))+"_"+file_name)
        model_orthogo=Orthogo(model)
        results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo=model_orthogo)



models_variant_archi_param(trainloader=trainloader,validloader=testloader,dataset="minicifar")