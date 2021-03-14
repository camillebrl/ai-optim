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
from binary_connect import BC
from regularisation import Orthogo
from pruning_thinet import Pruning
from train_validation import train_model, train_model_quantization
import constants as CN


def regularization(dataset, n_classes, train_loader, test_loader, n_epochs):
    logging.info("Regularizing model")
    model_name = "ResNet18"
    learning_rate = 0.001
    weight_decay = 5e-5
    momentum = 0.95
    loss_function = nn.CrossEntropyLoss()
    gradient_method = "SGD"
    scheduler = "CosineAnnealingLR"
    regul_coef = 0.2
    regul_function = "simple"

    regul_hparams = RegularizationHyperparameters(learning_rate, weight_decay,
                                                  momentum, loss_function,
                                                  gradient_method, model_name,
                                                  scheduler, regul_coef,
                                                  regul_function)
    model_nb_blocks = nums_blocks[model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)
    model.to(CN.DEVICE)
    optimizer, scheduler = get_optimizer_and_scheduler(regul_hparams,
                                                       model,
                                                       n_epochs)
    model_orthogo = Orthogo(model, CN.DEVICE)

    results = train_model(model, CN.DEVICE, loss_function,
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


def pruning(dataset, n_classes, train_loader, test_loader, n_epochs):
    listed_dir = f"./{dataset}/models/models_regularized"
    for f in os.listdir(listed_dir):
        logging.info(f"Pruning model in {f}")
        pruning_rate = 0.2
        pruning_type = "thinet_normal"
        model, hparams = load_model_and_hyperparameters(f, listed_dir, n_classes)

        pruning_hyperparameters = PruningHyperparameters(hparams.learning_rate,
                                                         hparams.weight_decay,
                                                         hparams.momentum,
                                                         hparams.loss_function,
                                                         hparams.gradient_method,
                                                         hparams.model_name,
                                                         hparams.scheduler,
                                                         pruning_rate,
                                                         pruning_type)
        optimizer, scheduler = get_optimizer_and_scheduler(hparams,
                                                           model,
                                                           n_epochs)

        model_pruned = Pruning(model, CN.DEVICE)

        if pruning_type == "thinet_normal":
            model_pruned.thinet(train_loader, pruning_rate)
        elif pruning_type == "thinet_batch":
            model_pruned.thinet_batch(train_loader, pruning_rate)

        results = train_model(model_pruned.model, CN.DEVICE, hparams.loss_function,
                              n_epochs, train_loader, test_loader, scheduler,
                              optimizer, None, None, None)

        fname = pruning_hyperparameters.build_name()
        model_dir = f"./{dataset}/models/models_pruned/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        logging.info("Saving model pruned" + fname)
        pruned_run = ModelRun(model.state_dict(), pruning_hyperparameters)
        torch.save(pruned_run, model_dir + fname_model)
        results.to_csv(results_dir + fname_results)


def quantization(dataset, n_classes, train_loader, test_loader, n_epochs):
    listed_dir = f"./{dataset}/models/models_pruned"
    for f in os.listdir(listed_dir):
        logging.info(f"Quantizing model in {f}")
        nb_bits = 3
        model, hparams = load_model_and_hyperparameters(f, listed_dir, n_classes)
        quanti_hparams = QuantizationHyperparameters(hparams.learning_rate,
                                                     hparams.weight_decay,
                                                     hparams.momentum,
                                                     hparams.loss_function,
                                                     hparams.gradient_method,
                                                     hparams.model_name,
                                                     hparams.scheduler,
                                                     nb_bits)

        optimizer, scheduler = get_optimizer_and_scheduler(hparams,
                                                           model,
                                                           n_epochs)
        bc_model = BC(model, nb_bits, CN.DEVICE)

        results = train_model_quantization(bc_model, CN.DEVICE,
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


def get_optimizer_and_scheduler(hyperparameters, model, n_epochs):
    if hyperparameters.gradient_method == "SGD":
        optimizer = optim.SGD(model.parameters(),
                              lr=hyperparameters.learning_rate,
                              momentum=hyperparameters.momentum,
                              weight_decay=hyperparameters.weight_decay)
    elif hyperparameters.gradient_method == "Adam":
        optimizer = optim.Adam(model.parameters(),
                               lr=hyperparameters.learning_rate,
                               momentum=hyperparameters.momentum,
                               weight_decay=hyperparameters.weight_decay)
    else:
        raise ValueError(
            f"Invalid optimizer found : {hyperparameters.optimizer} found,"
            f" expected Adam or SGD")

    if hyperparameters.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=n_epochs)
    elif hyperparameters.scheduler == "ReduceOnPlateau":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=n_epochs)
    else:
        raise ValueError(
            f"Invalid scheduler found : {hyperparameters.scheduler} found,"
            f" expected ReduceOnPlateau or CosineAnnealingLR")
    return optimizer, scheduler


def load_model_and_hyperparameters(filename, directory, n_classes):
    model_run = torch.load(os.path.join(directory, filename))
    hparams = model_run.hyperparameters
    model_nb_blocks = nums_blocks[hparams.model_name]
    model = ResNet(Bottleneck, model_nb_blocks, num_classes=n_classes)
    model.to(CN.DEVICE)
    model.load_state_dict(model_run.state_dict)
    return model, hparams
