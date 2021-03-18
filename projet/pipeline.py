import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from architecture_ResNet import ResNet, Bottleneck, nums_blocks
from hyperparameters import (RegularizationHyperparameters,
                             PruningHyperparameters,
                             QuantizationHyperparameters,
                             ClusteringHyperparameters)
from model_run import ModelRun
from binary_connect import BC
from regularisation import Orthogo
from pruning_thinet import Pruning
from train_validation import train_model, train_model_quantization, train_model_clustering
import constants as CN

# fonctions pour identifier certains éléments du fichier lu
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


def regularization(dataset, model_name, n_classes, train_loader, test_loader, n_epochs, regul_coef,
                   regul_function):
    """[summary]

    Args:
        model_name ([type]): "ResNet18", "ResNet34", "ResNet50", "ResNet101" ou "ResNet152"
        regul_function ([type]): "simple", "double", "mutual_coherence" ou "spectral_isometry"
    """
    logging.info("Regularizing model")
    # model_name = "ResNet18"
    learning_rate = 0.001
    weight_decay = 5e-5
    momentum = 0.95
    loss_function = nn.CrossEntropyLoss()
    gradient_method = "SGD"
    scheduler = "CosineAnnealingLR"

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
    tensorboard_logdir = regul_hparams.get_tensorboard_name()
    writer = SummaryWriter(os.path.join(CN.TBOARD, tensorboard_logdir))

    results = train_model(model, CN.DEVICE, loss_function,
                          n_epochs, train_loader, test_loader,
                          scheduler, optimizer, model_orthogo,
                          regul_function, regul_coef, writer)

    fname = regul_hparams.build_name()
    model_dir = f"./{dataset}/models/models_regularized/"
    results_dir = f"./{dataset}/results/"
    fname_model = fname + ".run"
    fname_results = fname + ".csv"
    logging.info("Saving model regularized" + fname)
    reqularized_run = ModelRun(model.state_dict(), regul_hparams)
    torch.save(reqularized_run, model_dir + fname_model)
    results.to_csv(results_dir + fname_results)
    writer.close()


def pruning(dataset, n_classes, train_loader, test_loader, n_epochs, pruning_rate, pruning_type):
    """[summary]

    Args:
        pruning_type ([type]): "thinet_normal" ou "thinet_batch"
    """
    listed_dir = f"./{dataset}/models/models_regularized"
    for f in os.listdir(listed_dir):
        logging.info(f"Pruning model in {f}")
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
        optimizer, scheduler = get_optimizer_and_scheduler(pruning_hyperparameters,
                                                           model,
                                                           n_epochs)
        tensorboard_logdir = pruning_hyperparameters.get_tensorboard_name()
        writer = SummaryWriter(os.path.join(CN.TBOARD, tensorboard_logdir))
        model_pruned = Pruning(model, CN.DEVICE)
        if pruning_type == "thinet_normal":
            model_pruned.thinet(train_loader, pruning_rate)
        elif pruning_type == "thinet_batch":
            model_pruned.thinet_batch(train_loader, pruning_rate)

        results = train_model(model_pruned.model, CN.DEVICE, hparams.loss_function,
                              n_epochs, train_loader, test_loader, scheduler,
                              optimizer, None, None, None, writer)
        regul_function = f[:findnth_right(f,"_",1)]
        regul_coefficient = f[findnth_left(f,"_",1):findnth_right(f,"_",2)]
        fname = pruning_hyperparameters.build_name()+"_"+regul_function+"_"+regul_coefficient
        model_dir = f"./{dataset}/models/models_pruned/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        logging.info("Saving model pruned" + fname)
        pruned_run = ModelRun(model.state_dict(), pruning_hyperparameters)
        torch.save(pruned_run, model_dir + fname_model)
        results.to_csv(results_dir + fname_results)
        writer.close()


def quantization(dataset, n_classes, train_loader, test_loader, n_epochs, nb_bits):
    listed_dir = f"./{dataset}/models/models_pruned"
    for f in os.listdir(listed_dir):
        logging.info(f"Quantizing model in {f}")
        model, hparams = load_model_and_hyperparameters(f, listed_dir, n_classes)
        quanti_hparams = QuantizationHyperparameters(hparams.learning_rate,
                                                     hparams.weight_decay,
                                                     hparams.momentum,
                                                     hparams.loss_function,
                                                     hparams.gradient_method,
                                                     hparams.model_name,
                                                     hparams.scheduler,
                                                     nb_bits)

        optimizer, scheduler = get_optimizer_and_scheduler(quanti_hparams,
                                                           model,
                                                           n_epochs)
        bc_model = BC(model, nb_bits, CN.DEVICE)

        tensorboard_logdir = quanti_hparams.get_tensorboard_name()
        writer = SummaryWriter(os.path.join(CN.TBOARD, tensorboard_logdir))
        results = train_model_quantization(bc_model, CN.DEVICE,
                                           hparams.loss_function, n_epochs,
                                           train_loader, test_loader,
                                           scheduler,
                                           optimizer,
                                           writer)

        pruning_type=f[findnth_left(f,"_",0):findnth_right(f,"_",2)]
        pruning_rate=f[findnth_left(f,"_",2):findnth_right(f,"_",3)]
        fname = quanti_hparams.build_name()+"_"+pruning_type+"_"+pruning_rate
        model_dir = f"./{dataset}/models/models_quantized/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        logging.info("Saving model quantized" + fname)
        quantized_run = ModelRun(model.state_dict(), quanti_hparams)
        torch.save(quantized_run, model_dir + fname_model)
        results.to_csv(results_dir + fname_results)


def clustering(dataset, n_classes, train_loader, test_loader, n_epochs, nb_clusters):
    listed_dir = f"./{dataset}/models/models_pruned"
    for f in os.listdir(listed_dir):
        logging.info(f"Clustering model in {f}")
        model, hparams = load_model_and_hyperparameters(f, listed_dir, n_classes)
        clustering_hparams = ClusteringHyperparameters(hparams.learning_rate,
                                                     hparams.weight_decay,
                                                     hparams.momentum,
                                                     hparams.loss_function,
                                                     hparams.gradient_method,
                                                     hparams.model_name,
                                                     hparams.scheduler,
                                                     nb_clusters)

        optimizer, scheduler = get_optimizer_and_scheduler(clustering_hparams,
                                                           model,
                                                           n_epochs)

        tensorboard_logdir = clustering_hparams.get_tensorboard_name()
        writer = SummaryWriter(os.path.join(CN.TBOARD, tensorboard_logdir))
        clustering_model, results = train_model_clustering(model, CN.DEVICE,
                                           hparams.loss_function, n_epochs,
                                           train_loader, test_loader,
                                           scheduler,
                                           optimizer,
                                           writer)

        pruning_type=f[findnth_left(f,"_",0):findnth_right(f,"_",2)]
        pruning_rate=f[findnth_left(f,"_",2):findnth_right(f,"_",3)]
        fname = clustering_hparams.build_name()+"_"+pruning_type+"_"+pruning_rate
        model_dir = f"./{dataset}/models/models_clustered/"
        results_dir = f"./{dataset}/results/"
        fname_model = fname + ".run"
        fname_results = fname + ".csv"
        logging.info("Saving model clustered" + fname)
        clustered_run = ModelRun(clustering_model.model.state_dict(), clustering_hparams)
        torch.save(clustered_run, model_dir + fname_model)
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
