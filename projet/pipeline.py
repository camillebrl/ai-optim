import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim

from architecture_ResNet import ResNet, Bottleneck, nums_blocks
from hyperparameters import (RegularizationHyperparameters,
                             PruningHyperparameters,
                             QuantizationHyperparameters)
from model_run import ModelRun
from regularisation import Orthogo
from pruning_thinet import Pruning
from train_validation import train_model, train_model_quantization
import constants as CN


def regularization(dataset, n_classes, train_loader, test_loader):
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
    regul_hparams = RegularizationHyperparameters(learning_rate, weight_decay,
                                                  momentum, loss_function,
                                                  gradient_method, model_name,
                                                  scheduler, regul_coef,
                                                  regul_function)
    model_nb_blocks = nums_blocks[model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)

    if gradient_method == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif gradient_method == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate,
                               momentum=momentum,
                               weight_decay=weight_decay)

    if scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=n_epochs)
    elif scheduler == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=n_epochs)

    model.to(device)
    model_orthogo = Orthogo(model, device)

    results = train_model(model, device, loss_function,
                          n_epochs, train_loader, test_loader,
                          scheduler, optimizer, model_orthogo,
                          regul_function, regul_coef)

    fname = regul_hparams.build_name()
    model_dir = f"./{dataset}/models/models_regularized/"
    results_dir = f"./{dataset}/results/"
    fname_model = fname + ".run"
    fname_results = fname + ".csv"
    logging.info("Saving model regularized" + fname)
    reqularized_run = ModelRun(model.state_dict(), regul_hparams)
    torch.save(reqularized_run, model_dir + fname_model)
    results.to_csv(results_dir + fname_results)


def quantization(dataset, n_classes, train_loader, test_loader):
    listed_dir = f"./{dataset}/models/models_pruned"
    for f in os.listdir(listed_dir):
        model_run = torch.load(os.path.join(listed_dir, f))
        hparams = model_run.hyperparameters

        device = CN.DEVICE
        model_nb_blocks = nums_blocks[hparams.model_name]
        model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)
        model.to(device)
        model.load_state_dict(model_run.state_dict)
        model.half()

        nb_bits = 3
        n_epochs = 200

        quanti_hparams = QuantizationHyperparameters(hparams.learning_rate,
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epochs)
        elif hparams.scheduler == "ReduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epochs)

        bc_model = BC(model, nb_bits, device)

        results = train_model_quantization(bc_model, device,
                                           hparams.loss_function, n_epochs,
                                           train_loader, test_loader,
                                           scheduler,
                                           optimizer)

        fname = quanti_hparams.build_name()
        model_dir = f"./{dataset}/models/models_quantized/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        logging.info("Saving model quantized" + fname)
        quantized_run = ModelRun(model.state_dict(), quanti_hparams)
        torch.save(quantized_run, model_dir + fname_model)
        results.to_csv(results_dir + fname_results)


def pruning(dataset, n_classes, train_loader, test_loader):
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
        n_epochs = 200

        pruning_hyperparameters = PruningHyperparameters(hparams.learning_rate,
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epochs)
        elif hparams.scheduler == "ReduceOnPlateau":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epochs)

        model_pruned = Pruning(model, device)

        if pruning_type == "thinet_normal":
            model_pruned.thinet(train_loader, pruning_rate)
        elif pruning_type == "thinet_batch":
            model_pruned.thinet_batch(train_loader, pruning_rate)

        results = train_model(model_pruned.model, device, hparams.loss_function,
                              n_epochs, train_loader, test_loader, scheduler,
                              optimizer, None, None, None)

        fname = pruning_hyperparameters.build_name()
        model_dir = f"./{dataset}/models/models_pruned/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        print("Saving model pruned" + fname)
        pruned_run = ModelRun(model.state_dict(), pruning_hyperparameters)
        torch.save(pruned_run, model_dir + fname_model)
        results.to_csv(results_dir + fname_results)
