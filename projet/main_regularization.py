import os

import torch
import torch.optim as optim
from binary_connect import BC
from train_validation import train, train_model, train_model_quantization, \
    validation, validation_half
from import_dataset import train_cifar10, train_cifar100, test_cifar10, \
    test_cifar100
import torch.nn as nn

from hyparameters import Regularization_Hyperparameters
from import_dataset import train_cifar10,train_cifar100,test_cifar10,test_cifar100
from architecture_ResNet import ResNet,Bottleneck,nums_blocks
import constants as CN
from regularisation import Orthogo


dataset="cifar10"
if dataset == "cifar10":
        trainloader = train_cifar10
        testloader = test_cifar10
elif dataset == "cifar100":
        trainloader = train_cifar100
        testloader = test_cifar100


model_name="ResNet18"
learning_rate=0.001
weight_decay=5e-5
momentum=0.95
loss_function=nn.CrossEntropyLoss()
gradient_method="SGD"
scheduler="CosineAnnealingLR"
n_epochs=200
regul_coef=0.2
regul_function="simple"

device = CN.DEVICE
regularization_Hyperparameters=Regularization_Hyperparameters(learning_rate,weight_decay,momentum,loss_function,gradient_method,model_name,scheduler,regul_coef,regul_function)
model_nb_blocks=nums_blocks[model_name]
model = ResNet(Bottleneck, model_nb_blocks, num_classes=int(dataset[dataset.find("1"):]))

if gradient_method == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum, weight_decay=weight_decay)
elif gradient_method == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,momentum=momentum, weight_decay=weight_decay)

if scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_epochs)
elif scheduler == "ReduceOnPlateau":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=n_epochs)

model.to(device)
model_orthogo = Orthogo(model, device)

results = train_model(model, device, loss_function,
                                          n_epochs, trainloader, testloader,
                                          scheduler, optimizer, model_orthogo,
                                          regul_function, regul_coef)

fname=regularization_Hyperparameters.build_name()
model_dir = f"./{dataset}/models/models_regularized/"
results_dir = f"./{dataset}/results/"
fname_model = fname+".pt"
fname_results = fname+".csv"
print("Saving model regularized"+fname)
torch.save(model.state_dict(),model_dir+fname_model)
results.to_csv(results_dir+fname_results)