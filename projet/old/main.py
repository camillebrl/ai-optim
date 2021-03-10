import os
import logging

from architecture_ResNet import nums_blocks, Bottleneck, ResNet
import torch
import torch.optim as optim
from regularisation import Orthogo
from pruning_thinet import Pruning
from binary_connect import BC
from train_validation import train, train_model, train_model_quantization, \
    validation, validation_half
from import_dataset import train_cifar10, train_cifar100, test_cifar10, \
    test_cifar100
import pandas as pd
import torch.nn as nn
import torch.backends.cudnn as cudnn

reg_coefs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
functions = ["simple", "spectral_isometry"]
pruning_rate = [0.2, 0.3, 0.4, 0.5, 0.6]
nb_bits_list = [i for i in range(2, 10)]
thinets = ["thinet_normal", "thinet_batch"]


def models_variant_archi_param(trainloader, validloader, dataset, n_epochs=150,
                               learning_rate=0.001, momentum=0.95,
                               weight_decay=5e-5, gradient_method="SGD",
                               scheduler="CosineAnnealingLR",
                               loss_function=nn.CrossEntropyLoss()):
    sched_name = scheduler
    for model_name, model_nb_blocks in zip(nums_blocks.keys(),
                                           nums_blocks.values()):
        for div_param in range(1, 9):
            model = ResNet(Bottleneck, model_nb_blocks, div=div_param,
                           num_classes=int(dataset[dataset.find("1"):]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # if device == 'cuda':
            #     model = torch.nn.DataParallel(model)
            #     cudnn.benchmark = True
            cudnn.benchmark = True
            model.to(torch.device("cuda:0"))
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                  momentum=momentum, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                   T_max=n_epochs)
            for function in functions:
                for reg_coef in reg_coefs:
                    model_orthogo = Orthogo(model, device)
                    results = train_model(model, device, loss_function,
                                          n_epochs, trainloader, validloader,
                                          scheduler, optimizer, model_orthogo,
                                          function, reg_coef)
                    fname_left = f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}"
                    fname_right = f"_lr_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradMethodOf_" \
                                  f"{gradient_method}_sched_{sched_name}"
                    for rate in pruning_rate:
                        model_pruning = Pruning(model, device)
                        for thinet in thinets:
                            if thinet == "thinet_normal":
                                model_pruning.thinet(trainloader, rate)
                            elif thinet == "thinet_batch":
                                model_pruning.thinet_batch(trainloader, rate)

                            base_dir = f"./{dataset}/model_{function}_reg_dif_para/"
                            model_dir = os.path.join(base_dir, "models")
                            results_dir = os.path.join(base_dir, "results")
                            print("model dir : ", model_dir)
                            print("results dir : ", results_dir)

                            # Results pruning
                            fname_pruning_base = fname_left + f"_ThiNet_pruning_rate_{rate}" + fname_right
                            fname_pruning_model = fname_pruning_base + ".pt"
                            fname_pruning_results = fname_pruning_base + ".csv"
                            print("Saving model in :", fname_pruning_model)
                            print("Saving results in :", fname_pruning_results)

                            results_pruning = {
                                "accuracy": [
                                    validation(n_epochs,
                                               model_pruning.model,
                                               device,
                                               validloader)
                                ]
                            }
                            results_pruning_df = pd.DataFrame.from_dict(results_pruning)
                            torch.save(model_pruning.model.state_dict(),

                                       os.path.join(model_dir, fname_pruning_model))
                            results_pruning_df.to_csv(
                                os.path.join(results_dir, fname_pruning_results))

                            # Results pruning retrained
                            pruning_retrain_name = f"_ThiNet_pruning_retrain_rate_{rate}"
                            fname_pruning_retrain_base = fname_left + pruning_retrain_name + fname_right
                            fname_pruning_retrain_model = fname_pruning_retrain_base + ".pt"
                            fname_pruning_retrain_results = fname_pruning_retrain_base + ".csv"
                            print("Saving model in :", fname_pruning_retrain_model)
                            print("Saving results in :", fname_pruning_retrain_results)

                            # TODO : IMPORTANT : USE THE VALIDATION RESULTS, THIS IS INCORRECT
                            results_pruning_retrained = train_model(
                                model_pruning.model, device, loss_function,
                                n_epochs, trainloader, validloader, scheduler,
                                optimizer, None, None, None)
                            torch.save(model_pruning.model.state_dict(),
                                       os.path.join(model_dir, fname_pruning_retrain_model))
                            results_pruning_retrained.to_csv(
                                os.path.join(results_dir, fname_pruning_retrain_results))

                            # Results pruning half precision
                            fname_pruning_retrain_half_base = "half_" + fname_pruning_retrain_base
                            fname_pruning_retrain_half_model = fname_pruning_retrain_half_base + ".pt"
                            fname_pruning_retrain_half_results = fname_pruning_retrain_half_base + ".csv"
                            print("Saving model in :", fname_pruning_retrain_half_model)
                            print("Saving results in :", fname_pruning_retrain_half_results)
                            model_pruned_half = model_pruning.model
                            model_pruned_half.half()
                            results_half_precision = {
                                "accuracy": [
                                    validation_half(n_epochs, model_pruned_half,
                                                    device,
                                                    validloader)
                                ]}
                            results_half_precision_df = pd.DataFrame.from_dict(
                                results_half_precision)
                            torch.save(model_pruned_half.state_dict(),
                                       os.path.join(model_dir, fname_pruning_retrain_half_model))
                            results_half_precision_df.to_csv(
                                os.path.join(results_dir, fname_pruning_retrain_half_results))

                            # Results pruning half precision with binary connect
                            for n_bits in nb_bits_list:
                                fname_pruning_retrain_half_bc_base = f"bc_{n_bits}" + fname_pruning_retrain_half_base
                                fname_pruning_retrain_half_bc_model = fname_pruning_retrain_half_bc_base + ".pt"
                                fname_pruning_retrain_half_bc_results = fname_pruning_retrain_half_bc_base + ".csv"
                                print("Saving model in :", fname_pruning_retrain_half_bc_model)
                                print("Saving results in :", fname_pruning_retrain_half_bc_results)

                                bc_model = BC(model_pruning.model, n_bits, device)
                                results_bc_model = train_model_quantization(
                                    bc_model, device, loss_function, n_epochs,
                                    trainloader, validloader, scheduler, optimizer)
                                torch.save(bc_model.model.state_dict(),
                                           os.path.join(model_dir,
                                                        fname_pruning_retrain_half_bc_model))
                                results_bc_model.to_csv(
                                    os.path.join(results_dir,
                                                 fname_pruning_retrain_half_bc_results))


for dataset in ["cifar10", "cifar100"]:
    if dataset == "cifar10":
        trainloader = train_cifar10
        testloader = test_cifar10
    elif dataset == "cifar100":
        trainloader = train_cifar100
        testloader = test_cifar100
    models_variant_archi_param(trainloader=trainloader,
                               validloader=testloader,
                               dataset=dataset)
