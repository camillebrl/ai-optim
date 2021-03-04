import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
import numpy as np
import torch.nn.utils
import torch.nn.functional as F

from resnet import ResNet, Bottleneck
from orthogonal import Orthogo
from pruning import Pruning

normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))

# Data augmentation is needed in order to train from scratch
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    # prend l'image de manière aléatoire et la tourne en mode miroir
    transforms.ToTensor(),
    # objigatoire pour pytorch: ne comprend que les tensors
    normalize_scratch,  # centre-réduit chaque tensor de l'image
])
# A noter: ici, on ne fait que changer la même image, on ne met pas différentes versions de l'image, ce n'est pas vraiment du data augmentation
# Il aurait fallu prendre le dataset, le multiplier, et appliquer cette transformation (flip, crop) sur la moitié du dataset par exemple

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])  # On ne fait pas le flip et le crop pour le test

### The data from CIFAR100 will be downloaded in the following dataset
rootdir_cifar100 = './data/cifar100'

c100train = CIFAR100(rootdir_cifar100, train=True, download=True,
                     transform=transform_train)
c100test = CIFAR100(rootdir_cifar100, train=False, download=True,
                    transform=transform_test)

train_cifar100 = DataLoader(c100train, batch_size=32)
test_cifar100 = DataLoader(c100test, batch_size=32)

### The data from CIFAR10 will be downloaded in the following dataset
rootdir_cifar10 = './data/cifar10'

c10train = CIFAR10(rootdir_cifar10, train=True, download=True,
                   transform=transform_train)
c10test = CIFAR10(rootdir_cifar10, train=False, download=True,
                  transform=transform_test)

train_cifar10 = DataLoader(c10train, batch_size=32)
test_cifar10 = DataLoader(c10test, batch_size=32)

######################################################################################################################################################
###################################################################VGG################################################################################
######################################################################################################################################################

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}

cfg_chosen = {}

for i in range(1, 9):
    model = []
    for x in cfg["VGG16"]:
        if x != "M":
            model.append(int(x / i))
        else:
            model.append(x)
    cfg_chosen[f"VGG16_{i}"] = model

nums_blocks = {"ResNet18": [2, 2, 2, 2], "ResNet34": [3, 4, 6, 3],
               "ResNet50": [3, 4, 6, 3], "ResNet101": [3, 4, 23, 3],
               "ResNet152": [3, 8, 36, 3]}


def function_separation_on_tensor(nb_bits, tensor, device):
    array_0 = np.linspace(start=-1, stop=1, num=2 ** nb_bits)
    full_array = np.stack([array_0] * np.prod(tensor.size())).reshape(
        tensor.size() + (
            array_0.shape))  # np.stack a une shape de taille a*b*c,4 (si tensor a une taille a*b*c). Nous on veut en sortie a,b,c,4 donc on reshape
    tensor_0 = torch.tensor(full_array, device=device)
    x = torch.unsqueeze(tensor, dim=-1)  # on transforme tensor en a,b,c,1
    # print(((tensor_0-x)**2).size())
    # print((torch.argmin((tensor_0-x)**2,dim=-1)))
    return torch.gather(tensor_0, dim=-1, index=torch.unsqueeze(
        torch.argmin((tensor_0 - x) ** 2, dim=-1),
        dim=-1))  # gather a besoin d'avoir les mêmes dimensions pour les 2
    # return tensor_0[torch.argmin((tensor_0-x)**2,dim=-1)] # on change le tensor en mettant dedans les valeurs les plus proches des valeurs à la même place de tensor_0


class BC():
    def __init__(self, model, nb_bits, device):

        # First we need to 
        # count the number of Conv2d and Linear
        # This will be used next in order to build a list of all 
        # parameters of the model 

        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 0
        end_range = count_targets - 1
        self.bin_range = np.linspace(start_range,
                                     end_range, end_range - start_range + 1) \
            .astype('int').tolist()

        # Now we can initialize the list of parameters

        self.num_of_params = len(self.bin_range)
        self.saved_params = []  # This will be used to save the full precision weights

        self.target_modules = []  # this will contain the list of modules to be modified

        self.model = model.half()  # this contains the model that will be trained and quantified

        self.device = device

        self.nb_bits = nb_bits

        ### This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:  # capable de binarizer certaines couches et pas d'autres
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):

        ### This loop goes through the list of target modules, and saves the corresponding weights into the list of saved_parameters
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        ### To be completed

        ### (1) Save the current full precision parameters using the save_params method
        self.save_params()
        ### (2) Binarize the weights in the model, by iterating through the list of target modules and overwrite the values with their binary version 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                function_separation_on_tensor(self.nb_bits,
                                              self.target_modules[index].data,
                                              self.device))
            # self.target_modules[index].cpu().detach().apply_(lambda x : -1 if x<0 else 1).cuda() # on ne peut pas appliquer la fonction apply_ avec gpu (uniquement sur cpu)

    def restore(self):

        ### restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):

        ## Clip all parameters to the range [-1,1] using Hard Tanh 
        ## you can use the nn.Hardtanh function
        hth = nn.Hardtanh()  # nn.Hardtanh est un Foncteur
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                hth(self.target_modules[index].data))
            # target_modules[index].data.copy_(hth(target_modules[index].detach().data)) # .data permet d'accéder aux données du tensor, et copy_ permet de faire inplace=True

    def forward(self, x):

        ### This function is used so that the model can be used while training
        out = self.model(x.half())

        return out


def train(epoch, model, optimizer, device, trainloader, loss_function,
          model_orthogo, function, reg_coef):
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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def train_quantization(epoch, bc_model, optimizer, device, trainloader,
                       loss_function):
    print('\nEpoch: %d' % epoch)
    bc_model.model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        bc_model.binarization()
        outputs = bc_model.forward(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        bc_model.restore()
        optimizer.step()
        bc_model.clip()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    acc = 100. * correct / total
    return acc


def validation(epoch, model, device, validloader):
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
    acc = 100. * correct / total
    return acc


def validation_quantization(epoch, bc_model, device, validloader):
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
    acc = 100. * correct / total
    return acc


def validation_half(epoch, model, device, validloader):
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
    acc = 100. * correct / total
    return acc


def get_schedulers(optimizer, n_epochs):
    schedulers = {
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs),
        "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max"),
    }
    return schedulers


def train_model(model, device, loss_function, n_epochs, trainloader,
                validloader, scheduler, optimizer, model_orthogo, function,
                reg_coef):
    best_acc_epoch = 0
    results = {"epoch": [], "train_accuracy": [], "validation_accuracy": []}
    epoch = 0
    dif = 0
    overfit_counter = 0
    previous_dif = 0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc = train(epoch, model, optimizer, device, trainloader,
                          loss_function, model_orthogo, function, reg_coef)
        valid_acc = validation(epoch, model, device, validloader)
        scheduler.step()
        print(train_acc, valid_acc)
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        dif = train_acc - valid_acc
        if dif > previous_dif:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_dif = dif
        epoch += 1
    return pd.DataFrame.from_dict(results).set_index("epoch")


def train_model_quantization(bc_model, device, loss_function, n_epochs,
                             trainloader, validloader, scheduler, optimizer):
    best_acc_epoch = 0
    results = {"epoch": [], "train_accuracy": [], "validation_accuracy": []}
    epoch = 0
    dif = 0
    overfit_counter = 0
    previous_dif = 0
    while epoch < n_epochs and overfit_counter < 10:
        train_acc = train_quantization(epoch, bc_model, optimizer, device,
                                       trainloader, loss_function)
        valid_acc = validation_quantization(epoch, bc_model, device,
                                            validloader)
        scheduler.step()  # il diminue le lr quand la validation accuracy n'augmente plus
        print(train_acc, valid_acc)
        results["train_accuracy"].append(train_acc)
        results["validation_accuracy"].append(valid_acc)
        results["epoch"].append(epoch)
        dif = train_acc - valid_acc
        if dif > previous_dif:
            overfit_counter += 1
        else:
            overfit_counter = 0
        if valid_acc > best_acc_epoch:
            best_acc_epoch = valid_acc
        previous_dif = dif
        epoch += 1
    return pd.DataFrame.from_dict(results).set_index("epoch")


reg_coefs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
functions = ["simple", "spectral_isometry"]
pruning_rate = [0.2, 0.3, 0.4, 0.5, 0.6]
nb_bits_list = [i for i in range(2, 10)]


def models_variant_archi_param(trainloader, validloader, n_epochs=1,
                               learning_rate=0.001, momentum=0.95,
                               weight_decay=5e-5, method_gradient_descent="SGD",
                               method_scheduler="CosineAnnealingLR",
                               loss_function=nn.CrossEntropyLoss(),
                               dataset="minicifar"):
    for model_name, model_nb_blocks in zip(nums_blocks.keys(),
                                           nums_blocks.values()):
        for div_param in range(1, 9):
            model = ResNet(Bottleneck, model_nb_blocks, div=div_param,
                           num_classes=int(dataset[dataset.find("1"):]))
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device == 'cuda':
                model = torch.nn.DataParallel(model)
                cudnn.benchmark = True
            optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                                  momentum=momentum, weight_decay=weight_decay)
            scheduler = get_schedulers(optimizer, n_epochs)[method_scheduler]
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
                        model_pruning.thinet(rate)
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


# Est-ce que j'ai besoin, à chaque fois que je modifie le modèle, de réinitialiser l'optimizer & le scheduler?

for dataset in ["cifar10", "cifar100"]:
    if dataset == "cifar10":
        trainloader = train_cifar10
        testloader = test_cifar10
    elif dataset == "cifar100":
        trainloader = train_cifar100
        testloader = test_cifar100
    models_variant_archi_param(trainloader=trainloader, validloader=testloader,
                               dataset=dataset)
