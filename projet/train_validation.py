import logging

import pandas as pd
import torch
from tqdm import tqdm

from weight_clustering import regul_deep_k_means, Cluster
from distillation import Distillation


def run_train_epoch(model, optimizer, device, trainloader, loss_function, model_orthogo, function,
                    reg_coef):
    model.train()
    epoch_loss = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        if function == "simple":
            loss += model_orthogo.soft_orthogonality_regularization(reg_coef)
        elif function == "double":
            loss += model_orthogo.double_soft_orthogonality_regularization(
                reg_coef)
        elif function == "mutual_coherence":
            loss += model_orthogo.mutual_coherence_orthogonality_regularization(
                reg_coef)
        elif function == "spectral_isometry":
            loss += model_orthogo.spectral_isometry_orthogonality_regularization(
                reg_coef)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
    epoch_loss /= len(trainloader)
    return epoch_loss


def run_train_epoch_quantization(bc_model, optimizer, device, trainloader, loss_function):
    bc_model.model.train()
    epoch_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc_model.binarization()
        outputs = bc_model.forward(inputs)
        loss = loss_function(outputs, targets)
        epoch_loss += loss.item()
        loss.backward()
        bc_model.restore()
        # TODO : log the loss to tensorboard
        optimizer.step()
        bc_model.clip()
        _, predicted = outputs.max(1)
    epoch_loss /= len(trainloader)
    return epoch_loss


def run_train_epoch_clustering(model, optimizer, device, trainloader, loss_function, regul_coef=0.0001):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss += regul_deep_k_means(model,device,regul_coef)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # TODO : log the loss to tensorboard
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def run_train_epoch_distillation_hilton(model_student, model_teacher, optimizer, device, trainloader, loss_function, regul_coef):
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_student = model_student(inputs)
        loss = loss_function(outputs_student, targets)
        outputs_teacher = model_teacher(inputs)
        loss += regul_coef*sum(outputs_teacher*torch.log(outputs_teacher/outputs_student))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # TODO : log the loss to tensorboard
        _, predicted = outputs_student.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc

def run_train_epoch_distillation_hilton2(model_student, model_teacher, optimizer, device, trainloader, loss_function, regul_coef):
    model_student.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs_student = model_student(inputs)
        loss = loss_function(outputs_student, targets)
        model_distil= Distillation(model_student,model_teacher,device)
        loss += model_distil.distillation_hilton(trainloader,regul_coef)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # TODO : log the loss to tensorboard
        _, predicted = outputs_student.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def run_validation_epoch(model, device, validloader, loss_function):
    model.eval()
    correct = 0
    total = 0
    epoch_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    epoch_loss /= len(validloader)
    return acc, epoch_loss


def run_validation_epoch_quantization(bc_model, device, validloader, loss_function):
    bc_model.model.eval()
    epoch_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bc_model.binarization()
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = bc_model.forward(inputs)
            # TODO : log the loss to tensorboard
            _, predicted = outputs.max(1)
            total += targets.size(0)
            loss = loss_function(outputs, targets)
            epoch_loss += loss.item()
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
    acc = 100. * correct / total
    epoch_loss /= len(validloader)
    return acc, epoch_loss


def train_model(model, device, loss_function, n_epochs, trainloader, validloader, scheduler,
                optimizer, model_orthogo, function, reg_coef, tboard):
    best_acc_epoch = 0
    results = {"epoch": [], "train_loss": [], "validation_accuracy": []}
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running normal training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_loss = run_train_epoch(model, optimizer, device, trainloader, loss_function,
                                     model_orthogo, function, reg_coef)
        valid_acc, valid_loss = run_validation_epoch(model, device, validloader, loss_function)
        scheduler.step()
        tboard.add_scalar("train/loss", train_loss, epoch)
        tboard.add_scalar("validation/loss", valid_loss, epoch)
        tboard.add_scalar("validation/accuracy", valid_acc, epoch)
        results["train_loss"].append(train_loss)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        loss_diff = valid_loss - train_loss
        if loss_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = loss_diff
    return pd.DataFrame.from_dict(results).set_index("epoch")


def train_model_quantization(bc_model, device, loss_function, n_epochs, trainloader, validloader,
                             scheduler, optimizer, tboard):
    best_acc_epoch = 0
    results = {"epoch": [], "train_loss": [], "validation_accuracy": []}
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running quantized training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_loss = run_train_epoch_quantization(bc_model, optimizer, device, trainloader,
                                                  loss_function)
        valid_acc, valid_loss = run_validation_epoch_quantization(bc_model, device, validloader,
                                                                  loss_function)
        # il diminue le lr quand la validation accuracy n'augmente plus
        scheduler.step()
        tboard.add_scalar("train/loss", train_loss, epoch)
        tboard.add_scalar("validation/loss", valid_loss, epoch)
        tboard.add_scalar("validation/accuracy", valid_acc, epoch)
        results["train_loss"].append(train_loss)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        loss_diff = valid_loss - train_loss
        if loss_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = loss_diff
    return pd.DataFrame.from_dict(results).set_index("epoch")


def train_model_clustering(model, device, loss_function, n_epochs, trainloader,
                validloader, scheduler, optimizer, nb_clusters):
    best_acc_epoch = 0
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running normal training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_acc = run_train_epoch_clustering(model, optimizer, device, trainloader, loss_function)
        valid_acc = run_validation_epoch(model, device, validloader)
        scheduler.step()
        accuracy_diff = train_acc - valid_acc
        if accuracy_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = accuracy_diff
    clustering_model=Cluster(model,nb_clusters,device)
    clustering_model.clustering()
    results = {"epoch": [], "validation_accuracy": []}
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        valid_acc = run_validation_epoch(clustering_model.model, device, validloader)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
    return clustering_model, pd.DataFrame.from_dict(results).set_index("epoch")