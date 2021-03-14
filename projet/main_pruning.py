import os
import logging

import torch
import torch.optim as optim
from train_validation import train_model

from architecture_ResNet import ResNet, Bottleneck, nums_blocks
from hyperparameters import PruningHyperparameters
from utils import init_logging, init_parser
from import_dataset import train_cifar10, train_cifar100, test_cifar10, test_cifar100
from pruning_thinet import Pruning
from model_run import ModelRun
import constants as CN

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

listed_dir = f"./{dataset}/models/models_regularized"

for f in os.listdir(listed_dir):
    logging.info(f"Pruning model in {f}")

    model_run = torch.load(os.path.join(listed_dir, f))
    hparams = model_run.hyperparameters

    device = CN.DEVICE
    model_nb_blocks = nums_blocks[hparams.model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)
    model.to(device)
    model.load_state_dict(model_run.state_dict)

    pruning_rate = 0.2
    pruning_type = "thinet_normal"
    pruning_function = "simple"
    n_epochs = 200

    pruning_Hyperparameters = PruningHyperparameters(hparams.learning_rate,
                                                     hparams.weight_decay,
                                                     hparams.momentum,
                                                     hparams.loss_function,
                                                     hparams.gradient_method,
                                                     hparams.model_name,
                                                     hparams.scheduler,
                                                     pruning_rate,
                                                     pruning_type)

    if hparams.gradient_method == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=hparams.learning_rate,
                              momentum=hparams.momentum,
                              weight_decay=hparams.weight_decay)
    elif hparams.gradient_method == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate,
                               momentum=hparams.momentum,
                               weight_decay=hparams.weight_decay)

    if hparams.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    elif hparams.scheduler == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    model_pruned = Pruning(model, device)

    if pruning_type == "thinet_normal":
        model_pruned.thinet(trainloader, pruning_rate)
    elif pruning_type == "thinet_batch":
        model_pruned.thinet_batch(trainloader, pruning_rate)

    results = train_model(model_pruned.model, device, hparams.loss_function,
                          n_epochs, trainloader, testloader, scheduler,
                          optimizer, None, None, None)

    fname = pruning_Hyperparameters.build_name()
    model_dir = f"./{dataset}/models/models_pruned/"
    results_dir = f"./{dataset}/results/"
    fname_model = fname + ".run"
    fname_results = fname + ".csv"
    print("Saving model pruned" + fname)
    pruned_run = ModelRun(model.state_dict(), pruning_Hyperparameters)
    torch.save(pruned_run, model_dir + fname_model)
    results.to_csv(results_dir + fname_results)
