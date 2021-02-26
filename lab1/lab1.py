import torch
import torch, torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms

import os
import argparse

from itertools import product


#############################################################################################
######################### Part1: familiarisation avec Pytorch ###############################
#############################################################################################

tensor = torch.rand(2, 2) # un block de 2 lignes avec 2 éléments par ligne
tensor[:1,:1] = 1 # 1ère ligne 1ère colonne
tensor[1:2,1:2] = 2 # deuxième ligne deuxième colonne
tensor[:1,1:2] = 3 # première ligne 2ème colonne
tensor[1:2,:1] = 4 # deuxième ligne 1ère colonne

tensor.mul(tensor) # met au carré chaque élément de tensor
tensor.matmul(tensor) # multiplie tensor par tensor (multiplication de matrices)
tensor.add_(4) # ajoute 4 à chaque élément de tensor

torch.rand(2,4,4,3) # 2 blocks de 4 dont chacun comporte 4 lignes et 3 éléments dans chaque ligne


model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

prediction = model(data)

# loss = (prediction - labels).sum()
# loss.backward() # backward pass

# optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
# optim.step()


# for i in data:
#     print(i)

# for i in prediction:
#     print(i)


# import torch

# a = torch.tensor([2., 3.], requires_grad=True)
# b = torch.tensor([6., 4.], requires_grad=True)




#############################################################################################
#################### Part2: VGG entraîné sur Minicifar avec Pytorch #########################
#############################################################################################

from minicifar import minicifar_train,minicifar_test,train_sampler,valid_sampler
from torch.utils.data.dataloader import DataLoader
trainloader= DataLoader(minicifar_train,batch_size=32,sampler=train_sampler)
validloader= DataLoader(minicifar_train,batch_size=32,sampler=valid_sampler)
testloader=DataLoader(minicifar_test,batch_size=32)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import pandas as pd




def print_score(estimator, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = estimator.predict(X_train)
        estimator_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{estimator_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")

        
    elif train==False:
        pred = estimator.predict(X_test)
        estimator_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{estimator_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")




net = torchvision.models.vgg16(pretrained=False)

# argument parser: interprète la ligne de commande; créé des commandes linux
# Ici, on dit juste à l'utilisateur qu'il peut fournir des arguments (--lr et --resume)
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training') # Permet d'interpréter les arguments
parser.add_argument('--lr', default=0.1, type=float, help='learning rate') # Ici choix du learning rate via terminal (de base 0.1)
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint') # chargement des poids via le fichier checkpoint; vaut True ou False.
parser.add_argument('--momentum',default=0.9,type=float,help='momentum')
parser.add_argument('--weight_decay',default=5e-4,type=float,help='weight decay') # weight decay: si les coefficients sont trop importants, ça exagère des différences ; si le features bouge un tout petit peu, ça change tout alors que le pattern n'est pas forcément significatif, du coup on fait de la régularisation/pénalisation des poids trop importants
args = parser.parse_args() # Créé le parser. Ensuite il suffit d'écrire args.lr et args.resume pour accéder aux valeurs des arguments

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume: # ajoute des arguments qui changent l'exécution du programme
    # Load checkpoint
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200) # learning rate qui descend
# lr_scheduler: comment on change le learning rate

# Training
def train(epoch,trainloader=trainloader, validloader=validloader,model=net,loss_function=nn.CrossEntropyLoss):
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
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100.*correct/total
    return acc


# Je ne vois pas en quoi la fonction test est liée à la fonction train? Normalement test doit reprendre le modèle (avec les poids) de train non?
def test(epoch,testloader=testloader,model=net,loss_function=nn.CrossEntropyLoss):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_function(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total

    # Ceci sauvegarde le meilleur modèle (l'époche qui a une accuracy meilleure que les autres époches que l'on a testé)

    # if acc > best_acc:
    #     print('Saving..')
    #     state = {
    #         'net': net.state_dict(),
    #         'acc': acc,
    #         'epoch': epoch,
    #     }
    #     if not os.path.isdir('checkpoint'):
    #         os.mkdir('checkpoint')
    #     torch.save(state, './checkpoint/ckpt.pth')
    #     best_acc = acc
    return acc    

# for epoch in range(start_epoch, start_epoch+200):
#     train(epoch)
#     test(epoch)
#     scheduler.step()


# def chose_scheduler_and_test(model,trainloader,validloader,testloader,optimizer,n_epochs,loss_function):
#     schedulers=[
#         torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs), #regarder la doc pour les différentes définitions
#         torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min"),
#         torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.001, max_lr=0.1),
#         torch.optim.lr_scheduler.LambdaLR(optimizer),
#         torch.optim.lr_scheduler.MultiplicativeLR(optimizer),
#         torch.optim.lr_scheduler.StepLR(optimizer),
#         torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,80]),
#         torch.optim.lr_scheduler.ExponentialLR(optimizer)
#     ]

#     for scheduler in schedulers:
#         for epoch in range(200):
#             train(epoch,loss_function,trainloader=trainloader,validloader=validloader,model=model) # Je ne comprends pas: ça fait juste train(0) ça non?
#             bast_acc=test(epoch,loss_function,testloader=testloader,model=model) # Et ça ça fait test(0) aussi non? Et pourquoi on met test ici??? 
#             scheduler.step() # Pourquoi on fait directement la rétropropagation de l'erreur? On n'a pas encore calculé l'erreur et fait "loss.backward()" pourtant...
# Pourquoi on met scheduler.step() à ce niveau? Dans train et test on a déjà une rétropropagation de l'erreur etc etc?
    

def gridsearch(model,trainloader,validloader,testloader,loss_function=nn.CrossEntropyLoss):
    learning_rates=[0.1,0.01,0.001]
    momentums=[0.85,0.9,0.95]
    weight_decay=[5e-5,5e-4,5e-3]
    method_gradient_descent=["SGD","ADAM"]
    n_epochs=200

    results={"learning_rate":[],"weight_decay":[],"method_gradient_descent":[],"momentum":[],"scheduler":[],"train_accuracy":[],"test_accuracy":[],"epoch":[]}

    for lr,wd,method in product(learning_rates,weight_decay,method_gradient_descent):
        print(lr,wd,method)
        if method=="SGD":
            for momentum in momentums:
                optimizer = optim.SGD(net.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=wd)
                schedulers=[
                    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs), #regarder la doc pour les différentes définitions
                    torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min"),
                    torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.001, max_lr=0.1),
                    torch.optim.lr_scheduler.StepLR(optimizer,step_size=30),
                    torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,80]),
                    torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.1)
                ]
                for scheduler in schedulers:
                    for epoch in range(200):
                        train_acc=train(epoch=epoch,loss_function=loss_function,trainloader=trainloader,validloader=validloader,model=model) # Je ne comprends pas: ça fait juste train(0) ça non?
                        test_acc=test(epoch,loss_function,testloader=testloader,model=model) # Et ça ça fait test(0) aussi non? Et pourquoi on met test ici??? 
                        scheduler.step() # Pourquoi on fait directement la rétropropagation de l'erreur? On n'a pas encore calculé l'erreur et fait "loss.backward()" pourtant...
                        # Pourquoi on met scheduler.step() à ce niveau? Dans train et test on a déjà une rétropropagation de l'erreur etc etc?
                        


                        results["learning_rate"].append(lr)
                        results["weight_decay"].append(wd)
                        results["method_gradient_descent"].append(method)
                        results["momentum"].append(momentum)
                        results["scheduler"].append(scheduler)
                        results["train_accuracy"].append(train_acc)
                        results["test_accuracy"].append(test_acc)
                        results["epoch"].append(epoch)

        elif method=="ADAM":
            optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=wd)
            schedulers=[
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs), #regarder la doc pour les différentes définitions
                torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="min"),
                torch.optim.lr_scheduler.CyclicLR(optimizer,base_lr=0.001, max_lr=0.1),
                torch.optim.lr_scheduler.LambdaLR(optimizer),
                torch.optim.lr_scheduler.MultiplicativeLR(optimizer),
                torch.optim.lr_scheduler.StepLR(optimizer),
                torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[30,80]),
                torch.optim.lr_scheduler.ExponentialLR(optimizer)
            ]
            for scheduler in schedulers:
                for epoch in range(200):
                    train(epoch=epoch,loss_function=loss_function,trainloader=trainloader,validloader=validloader,model=model) # Je ne comprends pas: ça fait juste train(0) ça non?
                    best_acc=test(epoch,loss_function,testloader=testloader,model=model) # Et ça ça fait test(0) aussi non? Et pourquoi on met test ici??? 
                    scheduler.step() # Pourquoi on fait directement la rétropropagation de l'erreur? On n'a pas encore calculé l'erreur et fait "loss.backward()" pourtant...
                    # Pourquoi on met scheduler.step() à ce niveau? Dans train et test on a déjà une rétropropagation de l'erreur etc etc?
                    results["learning_rate"].append(lr)
                    results["weight_decay"].append(wd)
                    results["method_gradient_descent"].append(method)
                    results["momentum"].append(np.nan)
                    results["scheduler"].append(scheduler)
                    results["accuracy"].append(best_acc)
                    results["epoch"].append(epoch)

    return pd.DataFrame.from_dict(results)


print(gridsearch(net,trainloader,validloader,testloader))