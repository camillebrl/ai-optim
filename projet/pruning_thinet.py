from architecture_ResNet import Bottleneck
import torch.nn as nn
import random
import torch
import numpy as np
import time
from tqdm import tqdm
import logging


def hook(module, input, output):
    if hasattr(module, "_input_hook"):
        # input contient plusieurs inputs différents?
        # ça ne fonctionne sans [0] car ça renvoie un tuple...
        module._input_hook = input[0]
        module._output_hook = output
    else:
        setattr(module, "_input_hook", input[0])
        setattr(module, "_output_hook", output)


class Pruning():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m, Bottleneck):
                convs = []
                for m_2 in m.modules():
                    if isinstance(m_2, nn.Conv2d):
                        convs.append(m_2)
                        # take all 2Dconv of each block except for the last one
                self.target_modules.extend(convs[:-1])

    def thinet(self, trainloader, pruning_rate):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                # register_forward_hook prend un objet fonction en paramètre
                m.register_forward_hook(hook)
        n = 64
        logging.info(f"Pruning on {len(self.target_modules)} modules with {n} batches")
        for mod, m in tqdm(enumerate(self.target_modules)):
            print("module",mod)
            if isinstance(m, nn.Conv2d):
                list_training = []
                # récupère au hasard les indices de n batch dans trainloader
                subset_indices = [random.randint(0, len(trainloader) - 1) for _
                                  in range(n)]
                for i, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if i in subset_indices:
                        for j in range(inputs.size()[0]):
                            size = inputs.size()
                            self.model(inputs[j].view((1, size[1], size[2], size[3])))
                            channel = random.randint(0, m._output_hook.size()[1] - 1)
                            ligne = random.randint(0, m._output_hook.size()[2] - 1)
                            colonne = random.randint(0, m._output_hook.size()[3] - 1)
                            # W = output_channel * input_channel * ligne * colonne
                            w = m.weight.data[channel, :, :, :]
                            # np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                            # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                            # x_2=torch.pad(m._input_hook[i][j,:,:,:],((0,0),(1,1),(1,1)))
                            output = m._input_hook[
                                0]  # on supprime la première dimension qu'on a ajouté précédemment
                            # pour redimensionner l'image car le modèle ne comprend que "liste d'image (cad batch)",
                            # dimension de chaque image
                            x_2 = torch.zeros((output.size()[0],
                                               output.size()[1] + 2,
                                               output.size()[2] + 2),
                                              device=self.device)
                            # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                            x_2[:, 1:-1, 1:-1] = m._input_hook[0]
                            # On ne prend pas -1 car le décalage est déjà là de base
                            x = x_2[:, ligne:ligne + w.size()[1],
                                colonne:colonne + w.size()[2]]
                            v = x * w
                            u = v.cpu().detach()  # v.cpu(): transforme torch.cuda.tensor en torch.tensor.
                            # Mais ça ne sort pas le tenseur du GPU, ça le transforme juste.
                            # Le .detach() transfère le tenseur sur le cpu et le sort du gpu.
                            list_training.append(u)
                            del x_2
                            del x
                            del w
                            del ligne
                            del colonne
                            del channel
                            del j
                            del u
                            del v
                            torch.cuda.empty_cache()
                    del inputs
                    del targets
                    torch.cuda.empty_cache()
                channels_to_delete = []
                total_channels = [i for i in range(m._input_hook.size()[1])]
                c = len(total_channels)
                while len(channels_to_delete) < c * pruning_rate:
                    min_value = np.inf
                    for channel in total_channels:
                        channels_to_try_to_delete = channels_to_delete + [
                            channel]
                        batch = torch.stack(list_training)
                        a_changed = batch[:, channels_to_try_to_delete, :,
                                    :]  # plus rapide pour pytorch de sommer un batch que de sommer une liste
                        result = torch.sum(a_changed,
                                           dim=(1, 2, 3))  # On ne somme pas sur la 1ère dimension 0
                        value = torch.sum(result ** 2)
                        if value < min_value:
                            min_value = value
                            min_channel = channel
                    channels_to_delete.append(min_channel)
                    total_channels.remove(min_channel)
                m.weight.data[:, channels_to_delete, :, :] = torch.zeros(
                    m.weight.data[:, channels_to_delete, :, :].size(),
                    device=self.device)
                del channels_to_try_to_delete
                del channels_to_delete
                del total_channels
                del result
                del value
                del min_value
                del m._input_hook
                del m._output_hook
                for a in list_training:
                    del a
                del list_training
                del batch
                del a_changed
                del min_channel
                torch.cuda.empty_cache()
                # print(time.time() - t0)
                # Pour simplifier, on ne supprime pas vraiment les poids à
                # enlever mais on les met à 0, car si on devait les supprimer,
                # il faudrait supprimer les channels en input aussi, et faire
                # une sorte de "backpropagation", ce qui est trop compliqué et
                # je n'ai pas le temps
                # Car total_channels ne contient que les poids que l'on garde
                # m.weight.data=m.weight.data[:,total_channels,:,:]
                # m._input_hook[i]=m._input_hook[i][total_channels,:,:]

    def thinet_batch(self, trainloader, p_to_delete):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                # register_forward_hook prend un objet fonction en paramètre
                m.register_forward_hook(hook)
        for mod, m in enumerate(self.target_modules):
            t0 = time.time()
            print("module:", mod)
            if isinstance(m, nn.Conv2d):
                list_training = []
                n = 64
                # récupère au hasard les indices de n batch dans trainloader
                subset_indices = [random.randint(0, len(trainloader) - 1) for _
                                  in range(n)]
                for i, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if i in subset_indices:
                        self.model(inputs)
                        batch_length = inputs.size()[0]
                        channel = random.randint(0, m._output_hook.size()[1] - 1)
                        ligne = random.randint(0, m._output_hook.size()[2] - 1)
                        colonne = random.randint(0, m._output_hook.size()[3] - 1)
                        # w = output_channel * input_channel * ligne * colonne
                        w = m.weight.data[channel, :, :, :]
                        output = m._input_hook
                        # np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                        # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                        # x_2=torch.pad(m._input_hook[i][j,:,:,:],((0,0),(1,1),(1,1)))
                        x_2 = torch.zeros((output.size()[0],
                                           output.size()[1],
                                           output.size()[2] + 2,
                                           output.size()[3] + 2),
                                          device=self.device)
                        # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                        # On s'intéresse sur la dimension des matrices, sur lesquelles on fait le padding
                        x_2[:, :, 1:-1, 1:-1] = output
                        # On ne prend pas -1 car le décalage est déjà là de base
                        x = x_2[:, :, ligne:ligne + w.size()[1],
                            colonne:colonne + w.size()[2]]
                        v = x * w
                        u = v.cpu().detach()  # v.cpu(): transforme torch.cuda.tensor en torch.tensor.
                        # Mais ça ne sort pas le tenseur du GPU, ça le transforme juste. 
                        # Le .detach() transfère le tenseur sur le cpu et le sort du gpu.
                        list_training.append(u)
                        del x_2
                        del x
                        del w
                        del ligne
                        del colonne
                        del channel
                        del u
                        del v
                        torch.cuda.empty_cache()
                    del inputs
                    del targets
                    torch.cuda.empty_cache()
                channels_to_delete = []
                total_channels = [i for i in range(m._input_hook.size()[1])]
                c = len(total_channels)
                while len(channels_to_delete) < c * p_to_delete:
                    min_value = np.inf
                    for channel in total_channels:
                        channels_to_try_to_delete = channels_to_delete + [
                            channel]
                        batch = torch.stack(list_training)
                        # sum on batch is better than sum on list
                        a_changed = batch[:, :, channels_to_try_to_delete, :, :]
                        # don't sum on first dimension 0
                        result = torch.sum(a_changed,
                                           dim=(2, 3, 4))
                        value = torch.sum(result ** 2)
                        if value < min_value:
                            min_value = value
                            min_channel = channel
                    channels_to_delete.append(min_channel)
                    total_channels.remove(min_channel)
                    zero_weight = torch.zeros(m.weight.data[:, channels_to_delete, :, :].size(),
                                              device=self.device)
                m.weight.data[:, channels_to_delete, :, :] = zero_weight
                del channels_to_try_to_delete
                del channels_to_delete
                del total_channels
                del result
                del value
                del min_value
                del m._input_hook
                del m._output_hook
                for a in list_training:
                    del a
                del list_training
                del batch
                del a_changed
                del min_channel
                torch.cuda.empty_cache()
                # Pour simplifier, on ne supprime pas vraiment les poids à
                # enlever mais on les met à 0, car si on devait les supprimer,
                # il faudrait supprimer les channels en input aussi, et faire
                # une sorte de "backpropagation", ce qui est trop compliqué et
                # je n'ai pas le temps
                # Car total_channels ne contient que les poids que l'on garde
                # m.weight.data=m.weight.data[:,total_channels,:,:]
                # m._input_hook[i]=m._input_hook[i][total_channels,:,:]
