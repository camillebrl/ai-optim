import os

import torch
import torch.optim as optim
from binary_connect import BC
from train_validation import train, train_model, train_model_quantization, \
    validation, validation_half
from import_dataset import train_cifar10, train_cifar100, test_cifar10, \
    test_cifar100
import torch.nn as nn

from hyparameters import Pruning_Hyperparameters
from import_dataset import train_cifar10,train_cifar100,test_cifar10,test_cifar100
from architecture_ResNet import ResNet,Bottleneck,nums_blocks
import constants as CN
from pruning_thinet import Pruning


def findnth_left(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)+1-len(parts[-1])-len(needle)

def findnth_right(haystack, needle, n):
    parts= haystack.split(needle, n+1)
    if len(parts)<=n+1:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


dataset="cifar10"
if dataset == "cifar10":
        trainloader = train_cifar10
        testloader = test_cifar10
elif dataset == "cifar100":
        trainloader = train_cifar100
        testloader = test_cifar100


for f in os.listdir(f"./{dataset}/models/models_regularized"):
    learning_rate=float(f[findnth_left(f,"_",4):findnth_right(f,"_",5)])
    weight_decay=float(f[findnth_left(f,"_",6):findnth_right(f,"_",7)])
    momentum=float(f[findnth_left(f,"_",5):findnth_right(f,"_",6)])
    if f[findnth_left(f,"_",3):findnth_right(f,"_",4)] == "CrossEntropyLoss()":
        loss_function=nn.CrossEntropyLoss()
    else:
        print("Erreur: Autre loss function que CrossEntropyLoss")
    gradient_method=f[findnth_left(f,"_",7):findnth_right(f,"_",8)]
    scheduler_name=f[findnth_left(f,"_",8):]
    if scheduler_name == "CosineAnnealingLR": 
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
    elif scheduler_name == "ReduceLROnPlateau": 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    model_name=f[findnth_left(f,"_",2):findnth_right(f,"_",3)]
    
    device = CN.DEVICE
    model_nb_blocks=nums_blocks[model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=int(dataset[dataset.find("1"):]))
    model.to(device)
    model.load_state_dict(torch.load(f"./{dataset}/models/models_regularized/{f}"))

    pruning_rate=0.2
    pruning_type="thinet_normal"
    pruning_function="simple"
    n_epochs=200


    if gradient_method == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum, weight_decay=weight_decay)
    elif gradient_method == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,momentum=momentum, weight_decay=weight_decay)

    
    pruning_Hyperparameters=Pruning_Hyperparameters(learning_rate,weight_decay,momentum,loss_function,gradient_method,model_name,scheduler,pruning_rate,pruning_type)
    
    model_pruned = Pruning(model, device)

    if pruning_type == "thinet_normal":
        model_pruned.thinet(trainloader, pruning_rate)
    elif pruning_type == "thinet_batch":
        model_pruned.thinet_batch(trainloader, pruning_rate)

    results=train_model(model_pruned.model, device, loss_function,
                        n_epochs, trainloader, testloader, scheduler,
                        optimizer, None, None, None)

    fname=pruning_Hyperparameters.build_name()
    model_dir = f"./{dataset}/models/models_pruned/"
    results_dir = f"./{dataset}/results/"
    fname_model = fname+".pt"
    fname_results = fname+".csv"
    print("Saving model pruned"+fname)
    torch.save(model_pruned.model.state_dict(),model_dir+fname_model)
    results.to_csv(results_dir+fname_results)