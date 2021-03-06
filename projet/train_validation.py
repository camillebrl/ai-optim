import pandas as pd
import torch

def train(epoch,model,optimizer,device,trainloader,loss_function,model_orthogo,function,reg_coef):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        if function == "simple":
            loss+=model_orthogo.soft_orthogonality_regularization(reg_coef)
        elif function == "double":
            loss+=model_orthogo.double_soft_orthogonality_regularization(reg_coef)
        elif function == "mutual_coherence":
            loss+=model_orthogo.mutual_coherence_orthogonality_regularization(reg_coef)
        elif function == "spectral_isometry":
            loss+=model_orthogo.spectral_isometry_orthogonality_regularization(reg_coef)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc



def train_quantization(epoch,bc_model,optimizer,device,trainloader,loss_function):
    print('\nEpoch: %d' % epoch)
    bc_model.model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc_model.binarization()
        outputs=bc_model.forward(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        bc_model.restore()
        optimizer.step()
        bc_model.clip()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc



def validation(epoch,model,device,validloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc    



def validation_quantization(epoch,bc_model,device,validloader):
    bc_model.model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        bc_model.binarization()
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = bc_model.forward(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    bc_model.restore()
    acc = 100.*correct/total
    return acc    


def validation_half(epoch,model,device,validloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.half())
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc    


def get_schedulers(optimizer,n_epochs):
    schedulers={
        "CosineAnnealingLR":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs),
        "ReduceLROnPlateau":torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max"),
    }
    return schedulers




def train_model(model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer,model_orthogo,function,reg_coef):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train(epoch,model,optimizer,device,trainloader,loss_function,model_orthogo,function,reg_coef) 
        valid_acc=validation(epoch,model,device,validloader)
        scheduler.step()
        print(train_acc,valid_acc)
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        dif=train_acc-valid_acc
        if dif > previous_dif:
            overfit_counter+=1
        else:
            overfit_counter=0
        if valid_acc > best_acc_epoch:
            best_acc_epoch=valid_acc
        previous_dif=dif
        epoch+=1
    return pd.DataFrame.from_dict(results).set_index("epoch")



def train_model_quantization(bc_model,device,loss_function,n_epochs,trainloader,validloader,scheduler,optimizer):
    best_acc_epoch=0
    results={"epoch":[],"train_accuracy":[],"validation_accuracy":[]}
    epoch=0
    dif=0
    overfit_counter=0
    previous_dif=0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc=train_quantization(epoch,bc_model,optimizer,device,trainloader,loss_function) 
        valid_acc=validation_quantization(epoch,bc_model,device,validloader)
        scheduler.step() # il diminue le lr quand la validation accuracy n'augmente plus
        print(train_acc,valid_acc)
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        dif=train_acc-valid_acc
        if dif > previous_dif:
            overfit_counter+=1
        else:
            overfit_counter=0
        if valid_acc > best_acc_epoch:
            best_acc_epoch=valid_acc
        previous_dif=dif
        epoch+=1
    return pd.DataFrame.from_dict(results).set_index("epoch")

