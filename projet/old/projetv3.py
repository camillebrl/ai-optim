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
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64//div, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128//div, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256//div, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512//div, num_blocks[3], stride=2)
        self.linear = nn.Linear((512//div)*block.expansion, num_classes)

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
###################################################################CountPara&FLOPS####################################################################
######################################################################################################################################################

def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/2 # FP16
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        m.register_buffer('total_params', torch.zeros(1))

        for p in m.parameters():
            m.total_params += torch.Tensor([p.numel()]) / 2 # Division Free quantification

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        total_params += m.total_params

    return total_ops, total_params

def count_param_and_flops(model,dataset):
    if dataset == "cifar10":
        # Resnet18 - Reference for CIFAR 10
        ref_params = 5586981
        ref_flops  = 834362880
    elif dataset == "cifar100":
        # WideResnet-28-10 - Reference for CIFAR 100
        ref_params = 36500000
        ref_flops  = 10490000000

    flops, params = profile(model, (1,3,32,32))
    flops, params = flops.item(), params.item()

    score_flops = flops / ref_flops
    score_params = params / ref_params
    score = score_flops + score_params
    print("Flops: {}, Params: {}".format(flops,params))
    print("Score flops: {} Score Params: {}".format(score_flops,score_params))
    print("Final score: {}".format(score))
    return flops, params, score_flops,score_params,score


######################################################################################################################################################
###################################################################BinaryConnect######################################################################
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




######################################################################################################################################################
###################################################################Regularisation#####################################################################
######################################################################################################################################################
class Orthogo():
    def __init__(self,model):
        self.model=model
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
            regul+=reg_coef_i*torch.norm(torch.transpose(w,0,1).matmul(w)-torch.eye(height,device="cuda:0"))**2
            #reg_grad=4*reg_coef*w*(torch.transpose(w,0,1)*w-torch.eye(height))
        return regul # le terme de régularisation est sur tous les modules! (somme)
            # bien régulariser les 1ères couches est interressant, et moins bien celles d'après. Donc le reg_coef doit être plus grand au début
            # plus reg_coef est grand, plus il sera régularisé. Donc, on diminue le reg_coef module après module
            # On peut tester de diviser reg_coef d'un module à l'autre (même méthodos que schedulers?)
    def double_soft_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.norm(torch.transpose(w,0,1).matmul(w)-torch.eye(height,device="cuda:0"))**2 + torch.norm(w.matmul(torch.transpose(w,0,1))-torch.eye(height,device="cuda:0"))**2)
        return regul
    def mutual_coherence_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.max(torch.abs((torch.transpose(w,0,1).matmul(w)-torch.eye(height,device="cuda:0"))**2)))
        return regul
    def spectral_isometry_orthogonality_regularization(self,reg_coef):
        regul=0
        for m in self.target_modules:
            width=np.prod(list(m.weight.data[0].size()))
            height=m.weight.data.size()[0]
            w=m.weight.data.view(width,height)
            regul+=reg_coef*(torch.nn.utils.spectral_norm((torch.transpose(w,0,1).matmul(w)-torch.eye(height,device="cuda:0"))))
        return regul

######################################################################################################################################################
###################################################################CountParameters####################################################################
######################################################################################################################################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    # p.numel: compte les éléments de p
    # requires_grad: pour déterminer les paramètres que le modèle peut apprendre (car ce sont ceux qui vont jouer dans la descente de gradient)

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


reg_coefs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
functions=["simple","double","mutual_coherence","spectral_isometry"]

def models_variant_archi_param(trainloader,validloader,n_epochs=2,learning_rate=0.001,momentum=0.95,weight_decay=5e-5,method_gradient_descent="SGD",method_scheduler="CosineAnnealingLR",loss_function=nn.CrossEntropyLoss(),dataset="minicifar"):
    for model_name,model_nb_blocks in zip(nums_blocks.keys(),nums_blocks.values()):
        for div_param in range(1,9):
            model=ResNet(Bottleneck,model_nb_blocks,div=div_param,num_classes=int(dataset[dataset.find("1"):]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=momentum, weight_decay=weight_decay)
            scheduler=get_schedulers(optimizer,n_epochs)[method_scheduler]
            for function in functions:
                for reg_coef in reg_coefs:
                    model_or=model
                    model_orthogo=Orthogo(model_or)
                    results=train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo,function,reg_coef)
                    flops,nb_para,flops_score,para_score,score=count_param_and_flops(model,dataset)
                    file_name=f"{model_name}_div_param_of_{div_param}_nb_param_of_{nb_para}_nb_flops_{flops}_scoreflops_{flops_score}_scorepara_{para_score}_score_{score}_function_of_{function}_reg_coef_of_{reg_coef}_{learning_rate}_{momentum}_{weight_decay}_{method_gradient_descent}_{method_scheduler}.csv"
                    results.to_csv(f"./{dataset}/model_{function}_reg_dif_para/"+file_name)

for dataset in ["cifar10","cifar100"]:

    if dataset == "cifar10":
        trainloader=train_cifar10
        testloader=test_cifar10
    elif dataset == "cifar100":
        trainloader=train_cifar100
        testloader=test_cifar100

    models_variant_archi_param(trainloader=trainloader,validloader=testloader,dataset=dataset)