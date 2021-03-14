import os

import torch
import torch.optim as optim
from binary_connect import BC
from train_validation import train_model_quantization
import torch.nn as nn

from architecture_ResNet import ResNet, Bottleneck, nums_blocks
from hyperparameters import QuantizationHyperparameters
from import_dataset import train_cifar10, train_cifar100, test_cifar10, test_cifar100
from utils import init_logging, init_parser
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

for f in os.listdir(f"./{dataset}/models/models_pruned"):
    model_run = torch.load(f"./{dataset}/models/models_pruned/{f}")
    hparams = model_run.hyperparameters

    device = CN.DEVICE
    model_nb_blocks = nums_blocks[hparams.model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)
    model.to(device)
    model.load_state_dict(model_run.state_dict)
    model.half()

    nb_bits = 3
    n_epochs = 200

    quantization_Hyperparameters = QuantizationHyperparameters(hparams.learning_rate,
                                                               hparams.weight_decay,
                                                               hparams.momentum,
                                                               hparams.loss_function,
                                                               hparams.gradient_method,
                                                               hparams.model_name,
                                                               hparams.scheduler,
                                                               hparams.nb_bits)

    if hparams.gradient_method == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=hparams.learning_rate,
                              momentum=hparams.momentum,
                              weight_decay=hparams.weight_decay)
    elif hparams.gradient_method == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=hparams.learning_rate,
                               momentum=hparams.momentum,
                               weight_decay=hparams.weight_decay)
    if hparams.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hparams.optimizer, T_max=n_epochs)
    elif hparams.scheduler == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(hparams.optimizer, T_max=n_epochs)

    bc_model = BC(model, nb_bits, device)

    results = train_model_quantization(bc_model, device, hparams.loss_function, n_epochs,
                                       trainloader, testloader, hparams.scheduler,
                                       hparams.optimizer)

    fname = quantization_Hyperparameters.build_name()
    model_dir = f"./{dataset}/models/models_quantized/"
    results_dir = f"./{dataset}/results/"
    fname_model = fname + ".run"
    fname_results = fname + ".csv"
    logging.info("Saving model quantized" + fname)
    quantized_run = ModelRun(model.state_dict(), quantization_Hyperparameters)
    torch.save(quantized_run, model_dir + fname_model)
    results.to_csv(results_dir + fname_results)
