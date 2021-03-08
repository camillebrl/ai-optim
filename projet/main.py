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
thinets = ["thinet_normal","thinet_batch"]

def models_variant_archi_param(trainloader, validloader, dataset, n_epochs=150,
                               learning_rate=0.001, momentum=0.95,
                               weight_decay=5e-5, method_gradient_descent="SGD",
                               method_scheduler="CosineAnnealingLR",
                               loss_function=nn.CrossEntropyLoss()):
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
                    file_name = f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv"
                    for rate in pruning_rate:
                        model_pruning = Pruning(model, device)
                        for thinet in thinets:
                            if thinet == "thinet_normal":
                                model_pruning.thinet(trainloader, rate)
                            elif thinet == "thinet_batch":
                                model_pruning.thinet_batch(trainloader, rate)
                            file_name = f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv"
                            results_pruning = {"accuracy": []}
                            results_pruning["accuracy"].append(
                                validation(n_epochs, model_pruning.model, device,
                                        validloader))
                            results_pruning_df = pd.DataFrame.from_dict(
                                results_pruning)
                            torch.save(model_pruning.model.state_dict(),
                                    f"./{dataset}/model_{function}_reg_dif_para/models/" + f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                            results_pruning_df.to_csv(
                                f"./{dataset}/model_{function}_reg_dif_para/results/" + f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                            results_pruning_retrained = train_model(
                                model_pruning.model, device, loss_function,
                                n_epochs, trainloader, validloader, scheduler,
                                optimizer, None, None, None)
                            torch.save(model_pruning.model.state_dict(),
                                    f"./{dataset}/model_{function}_reg_dif_para/models/" + f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                            results_pruning_retrained.to_csv(
                                f"./{dataset}/model_{function}_reg_dif_para/results/" + f"{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                            model_pruned_half = model_pruning.model
                            model_pruned_half.half()
                            results_half_precision = {"accuracy": []}
                            results_half_precision["accuracy"].append(
                                validation_half(n_epochs, model_pruned_half, device,
                                                validloader))
                            results_half_precision_df = pd.DataFrame.from_dict(
                                results_half_precision)
                            torch.save(model_pruned_half.state_dict(),
                                    f"./{dataset}/model_{function}_reg_dif_para/models/" + f"half_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                            results_half_precision_df.to_csv(
                                f"./{dataset}/model_{function}_reg_dif_para/results/" + f"half_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")
                            for nb_bits in nb_bits_list:
                                bc_model = BC(model_pruning.model, nb_bits, device)
                                results_bc_model = train_model_quantization(
                                    bc_model, device, loss_function, n_epochs,
                                    trainloader, validloader, scheduler, optimizer)
                                torch.save(bc_model.model.state_dict(),
                                        f"./{dataset}/model_{function}_reg_dif_para/models/" + f"half_quantized_with_BinaryConnect_precisionof_{nb_bits}bits_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.pt")
                                results_bc_model.to_csv(
                                    f"./{dataset}/model_{function}_reg_dif_para/results/" + f"half_quantized_with_BinaryConnect_precisionof_{nb_bits}bits_{model_name}_divParamOf_{div_param}_functionOf_{function}_regCoefOf_{reg_coef}_ThiNet_pruning_retrain_rate_{rate}_learningRateOf_{learning_rate}_momentumOf_{momentum}_weightDecayOf_{weight_decay}_gradDescentMethodOf_{method_gradient_descent}_schedMethodOf_{method_scheduler}.csv")


# Est-ce que j'ai besoin, à chaque fois que je modifie le modèle,
# de réinitialiser l'optimizer & le scheduler?


for dataset in ["cifar10", "cifar100"]:
    if dataset == "cifar10":
        trainloader = train_cifar10
        testloader = test_cifar10
    elif dataset == "cifar100":
        trainloader = train_cifar100
        testloader = test_cifar100
    models_variant_archi_param(trainloader=trainloader, validloader=testloader,
                               dataset=dataset)
