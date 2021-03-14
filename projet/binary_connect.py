import torch
import torch.nn as nn
import numpy as np
import time


def function_separation_on_tensor(nb_bits, tensor, device):
    array_0 = np.linspace(start=-1, stop=1, num=2 ** nb_bits)
    # np.stack a une shape de taille a*b*c,4 (si tensor a une taille a*b*c).
    # Nous on veut en sortie a,b,c,4 donc on reshape
    full_array = np.stack([array_0] * np.prod(tensor.size())).reshape(
        tensor.size() + (array_0.shape))
    tensor_0 = torch.tensor(full_array, device=device)
    x = torch.unsqueeze(tensor, dim=-1)  # on transforme tensor en a,b,c,1
    # on change le tensor en mettant dedans les valeurs les plus proches des valeurs à la même place de tensor_0
    results = torch.gather(tensor_0, dim=-1, index=torch.unsqueeze(
        torch.argmin((tensor_0 - x) ** 2, dim=-1), dim=-1))
    # gather a besoin d'avoir les mêmes dimensions pour les 2
    return results.view(
        tensor.size())  # on remet de la même taille que le tensor initial vu qu'on avait ajouté une dim
    # return tensor_0[torch.argmin((tensor_0-x)**2,dim=-1)]


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
        # This will be used to save the full precision weights
        self.saved_params = []

        # this will contain the list of modules to be modified
        self.target_modules = []

        # this contains the model that will be trained and quantified
        self.model = model.half()

        self.device = device

        self.nb_bits = nb_bits

        # This builds the initial copy of all parameters and target modules
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                # capable de binarizer certaines couches et pas d'autres
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def save_params(self):

        # This loop goes through the list of target modules, and saves the
        # corresponding weights into the list of saved_parameters
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)

    def binarization(self):

        # To be completed

        # (1) Save the current full precision parameters using the save_params method
        self.save_params()
        # (2) Binarize the weights in the model, by iterating through the list
        # of target modules and overwrite the values with their binary version
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                function_separation_on_tensor(self.nb_bits,
                                              self.target_modules[index].data,
                                              self.device))
        # on ne peut pas appliquer la fonction apply_ avec gpu (uniquement sur cpu)
        # self.target_modules[index].cpu().detach().apply_(
        #     lambda x: -1 if x < 0 else 1).cuda()

    def restore(self):

        # restore the copy from self.saved_params into the model 
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def clip(self):

        # Clip all parameters to the range [-1,1] using Hard Tanh
        # you can use the nn.Hardtanh function
        hth = nn.Hardtanh()  # nn.Hardtanh est un Foncteur
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                hth(self.target_modules[index].data))
            # target_modules[index].data.copy_(hth(target_modules[index].detach().data))
            # .data permet d'accéder aux données du tensor, et copy_ permet de faire inplace=True

    def forward(self, x):

        # This function is used so that the model can be used while training
        out = self.model(x.half())

        return out
