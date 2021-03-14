import logging
import torch
import torch.optim as optim
from train_validation import train_model
import torch.nn as nn

from hyperparameters import RegularizationHyperparameters
from import_dataset import train_cifar10, train_cifar100, test_cifar10, test_cifar100
from architecture_ResNet import ResNet, Bottleneck, nums_blocks
import constants as CN
from regularisation import Orthogo
from model_run import ModelRun
from utils import init_logging, init_parser

init_logging()
parser = init_parser()
args = parser.parse_args()
dataset = args.dataset
if dataset == "cifar10":
    n_classes = 10
    trainloader = train_cifar10
    testloader = test_cifar10
elif dataset == "cifar100":
    n_classes = 100
    trainloader = train_cifar100
    testloader = test_cifar100

model_name = "ResNet18"
learning_rate = 0.001
weight_decay = 5e-5
momentum = 0.95
loss_function = nn.CrossEntropyLoss()
gradient_method = "SGD"
scheduler = "CosineAnnealingLR"
n_epochs = 200
regul_coef = 0.2
regul_function = "simple"

device = CN.DEVICE
regularization_Hyperparameters = RegularizationHyperparameters(learning_rate, weight_decay,
                                                               momentum, loss_function,
                                                               gradient_method, model_name,
                                                               scheduler, regul_coef,
                                                               regul_function)
model_nb_blocks = nums_blocks[model_name]
model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)

if gradient_method == "SGD":
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum,
                          weight_decay=weight_decay)
elif gradient_method == "Adam":
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, momentum=momentum,
                           weight_decay=weight_decay)

if scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
elif scheduler == "ReduceOnPlateau":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

model.to(device)
model_orthogo = Orthogo(model, device)

results = train_model(model, device, loss_function,
                      n_epochs, trainloader, testloader,
                      scheduler, optimizer, model_orthogo,
                      regul_function, regul_coef)

fname = regularization_Hyperparameters.build_name()
model_dir = f"./{dataset}/models/models_regularized/"
results_dir = f"./{dataset}/results/"
fname_model = fname + ".run"
fname_results = fname + ".csv"
logging.info("Saving model regularized" + fname)
reqularized_run = ModelRun(model.state_dict(), regularization_Hyperparameters)
torch.save(reqularized_run, model_dir + fname_model)
results.to_csv(results_dir + fname_results)
