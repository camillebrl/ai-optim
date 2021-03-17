import logging

import pandas as pd
import torch
from tqdm import tqdm

from weight_clustering import regul_deep_k_means, Cluster
from distillation import Distillation


def run_train_epoch(model, optimizer, device, trainloader, loss_function, model_orthogo, function,
                    reg_coef):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
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
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # TODO : log the loss to tensorboard
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def run_train_epoch_quantization(bc_model, optimizer, device, trainloader, loss_function):
    bc_model.model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc_model.binarization()
        outputs = bc_model.forward(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        bc_model.restore()
        # TODO : log the loss to tensorboard
        optimizer.step()
        bc_model.clip()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def run_train_epoch_clustering(model, optimizer, device, trainloader, loss_function, function, reg_coef):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss += regul_deep_k_means(model, reg_coef, device)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # TODO : log the loss to tensorboard
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    clustering_model=Cluster(model)
    clustering_model.clustering()
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



def run_validation_epoch(model, device, validloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            # TODO : log the loss to tensorboard
            correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def run_validation_epoch_quantization(bc_model, device, validloader):
    bc_model.model.eval()
    test_loss = 0
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
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
    acc = 100. * correct / total
    return acc



def train_model(model, device, loss_function, n_epochs, trainloader,
                validloader, scheduler, optimizer, model_orthogo, function,
                reg_coef):
    best_acc_epoch = 0
    results = {"epoch": [], "train_accuracy": [], "validation_accuracy": []}
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running normal training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_acc = run_train_epoch(model, optimizer, device, trainloader, loss_function,
                                    model_orthogo, function, reg_coef)
        valid_acc = run_validation_epoch(model, device, validloader)
        scheduler.step()
        # TODO : use tensorboard
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        accuracy_diff = train_acc - valid_acc
        if accuracy_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = accuracy_diff
    return pd.DataFrame.from_dict(results).set_index("epoch")


def train_model_quantization(bc_model, device, loss_function, n_epochs,
                             trainloader, validloader, scheduler, optimizer):
    best_acc_epoch = 0
    results = {"epoch": [], "train_accuracy": [], "validation_accuracy": []}
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running quantized training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_acc = run_train_epoch_quantization(bc_model, optimizer, device, trainloader,
                                                 loss_function)
        valid_acc = run_validation_epoch_quantization(bc_model, device, validloader)
        # il diminue le lr quand la validation accuracy n'augmente plus
        scheduler.step()
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        accuracy_diff = train_acc - valid_acc
        if accuracy_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = accuracy_diff
    return pd.DataFrame.from_dict(results).set_index("epoch")

def train_model_clustering(model, device, loss_function, n_epochs, trainloader,
                validloader, scheduler, optimizer, model_orthogo, function,
                reg_coef):
    best_acc_epoch = 0
    results = {"epoch": [], "train_accuracy": [], "validation_accuracy": []}
    overfit_counter = 0
    previous_diff = 0
    logging.info(f"Running normal training on {n_epochs} epochs")
    for epoch in tqdm(range(1, n_epochs + 1)):
        if overfit_counter > 10:
            logging.info(f"Early stopping at epoch {epoch}")
            break
        train_acc = run_train_epoch_clustering(model, optimizer, device, trainloader, loss_function,
                                    model_orthogo, function, reg_coef)
        valid_acc = run_validation_epoch(model, device, validloader)
        scheduler.step()
        # TODO : use tensorboard
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        accuracy_diff = train_acc - valid_acc
        if accuracy_diff > previous_diff:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_diff = accuracy_diff
    return pd.DataFrame.from_dict(results).set_index("epoch")