from architecture_ResNet import Bottleneck
import torch.nn as nn
import random
import torch
import numpy as np

def hook(module, input, output):
    if hasattr(module,"_input_hook"):
        module._input_hook=input[0] # input contient plusieurs inputs différents? ça ne fonctionne sans [0] car ça renvoie un tuple...
        module._output_hook=output
    else:
        setattr(module,"_input_hook",input[0])
        setattr(module,"_output_hook",output)

class Pruning():
    def __init__(self,model,device):
        self.model=model
        self.device=device
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m,Bottleneck):
                convs=[]
                for m_2 in m.modules():
                    if isinstance(m_2,nn.Conv2d):
                        convs.append(m_2)
                self.target_modules.extend(convs[:-1]) # on prend toutes les conv2d de chaque block sauf la dernière
        print(self.target_modules)
    def thinet(self,trainloader,p_to_delete):
        for m in self.target_modules:
            if isinstance(m, nn.Conv2d):
                m.register_forward_hook(hook) # register_forward_hook prend un objet fonction en paramètre
        for mod,m in enumerate(self.target_modules): # Dans le papier, il est indiqué que 90% des floating points operattions sont contenus dans les 10 premiers layers
            print("module:",mod)
            if isinstance(m, nn.Conv2d):    
                list_training=[]
                n=64
                subset_indices = [random.randint(0,len(trainloader)-1) for _ in range(n)] # récupère au hasard les indices de n batch dans trainloader
                for i, (inputs, targets) in enumerate(trainloader):
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    if i in subset_indices:
                        for j in range(inputs.size()[0]):
                            size=inputs.size()
                            self.model(inputs[j].view((1,size[1],size[2],size[3])))
                            channel=random.randint(0,m._output_hook.size()[1]-1)
                            ligne=random.randint(0,m._output_hook.size()[2]-1)
                            colonne=random.randint(0,m._output_hook.size()[3]-1)
                            w=m.weight.data[channel,:,:,:] # W = output_channel * input_channel * ligne * colonne
                            #np.pad pour ajouter des 0 sur un objet de type numpy, mais pas compatible avec tensor!
                            #x_2=torch.pad(m._input_hook[i][j,:,:,:],((0,0),(1,1),(1,1))) # premier tuple: pour ajouter sur la dim channel, 2ème sur la dim ligne, 3ème sur dim colonne
                            x_2=torch.zeros((m._input_hook[0].size()[0],m._input_hook[0].size()[1]+2,m._input_hook[0].size()[2]+2),device=self.device)
                            x_2[:,1:-1,1:-1] = m._input_hook[0] # On remplace une matrice avec que des 0 avec nos valeurs de x à l'intérieur (padding autour)
                            x=x_2[:,ligne:ligne+w.size()[1],colonne:colonne+w.size()[2]] # On ne prend pas -1 car le décalage est déjà là de base
                            list_training.append(x*w)       
                channels_to_delete=[]
                channels_to_try_to_delete=[]
                total_channels=[i for i in range(m._input_hook.size()[1])]
                c=len(total_channels)
                while len(channels_to_delete)<c*p_to_delete:
                    min_value=np.inf
                    for channel in total_channels:
                        channels_to_try_to_delete=channels_to_delete+[channel]
                        value=0
                        for a in list_training:
                            a_changed=a[channels_to_try_to_delete,:,:]
                            result=torch.sum(a_changed)
                            value+=result**2
                        if value<min_value:
                            min_value=value
                            min_channel=channel
                    channels_to_delete.append(min_channel)
                    total_channels.remove(min_channel)
                m.weight.data[:,channels_to_delete,:,:]=torch.zeros(m.weight.data[:,channels_to_delete,:,:].size(),device=self.device)
                #Pour simplifier, on ne supprime pas vraiment les poids à enlever mais on les met à 0, car si on devait les supprimer, il faudrait supprimer les channels en input aussi, et faire une sorte de "backpropagation", ce qui est trop compliqué et je n'ai pas le temps
                #m.weight.data=m.weight.data[:,total_channels,:,:] # Car total_channels ne contient que les poids que l'on garde
                #m._input_hook[i]=m._input_hook[i][total_channels,:,:]

