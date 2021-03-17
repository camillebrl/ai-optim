import os
import torch
import torch.nn as nn
import logging
from pipeline import load_model_and_hyperparameters

def count_conv2d(m, x, y):
    x = x[0] # remove tuple

    fin = m.in_channels
    fout = m.out_channels
    sh, sw = m.kernel_size

    # ops per output element
    kernel_mul = sh * sw * fin
    kernel_add = sh * sw * fin - 1
    bias_ops = 1 if m.bias is not None else 0
    kernel_mul = kernel_mul/2 # FP16
    ops = kernel_mul + kernel_add + bias_ops

    # total ops
    num_out_elements = y.numel()
    total_ops = num_out_elements * ops

    print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, params={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple

    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div

    m.total_ops += torch.Tensor([int(total_ops)])
    print("Batch norm: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(m.total_params.item()),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]

    nelements = x.numel()
    total_ops = nelements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, params={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, params={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    # per output element
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    print("Linear: F_in={}, F_out={}, params={}, operations={}".format(m.in_features,m.out_features,int(m.total_params.item()),int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")

# custom ops could be used to pass variable customized ratios for quantization
def profile(model, input_size, custom_ops = {}):

    model.eval()

    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))
        # m.register_buffer('total_params', torch.zeros(1))

        # for p in m.parameters():
        #     m.total_params += torch.Tensor([p.numel()]) / 2 # Division Free quantification

        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(count_conv2d)
        elif isinstance(m, nn.BatchNorm2d):
            m.register_forward_hook(count_bn2d)
        elif isinstance(m, nn.ReLU):
            m.register_forward_hook(count_relu)
        elif isinstance(m, (nn.AvgPool2d)):
            m.register_forward_hook(count_avgpool)
        elif isinstance(m, nn.Linear):
            m.register_forward_hook(count_linear)
        elif isinstance(m, nn.Sequential):
            m.register_forward_hook(count_sequential)
        else:
            print("Not implemented for ", m)

    model.apply(add_hooks)

    x = torch.zeros(input_size)
    model(x)

    total_ops = 0
    # total_params = 0
    for m in model.modules():
        if len(list(m.children())) > 0: continue
        total_ops += m.total_ops
        # total_params += m.total_params

    return total_ops
    # return total_ops, total_params


#################################################
################# IMPORTANT #####################
#################################################
# Il se peut que ça ne foncitonne pas le if (isinstance(m,nn.qqch));
# Effectivement, on a des bottelneck avec resnet!
# for m in self.model.modules():
#     if isinstance(m, Bottleneck):
#         convs = []
#         for m_2 in m.modules():
#             if isinstance(m_2, nn.Conv2d):
#                 convs.append(m_2)
#                 # take all 2Dconv of each block except for the last one
#         self.target_modules.extend(convs[:-1])


def num_of_params(model):
    nb_params=model.numel()
    for param in model.parameters():
        # count number of 0, sum the total and remove it
        if param is not None :
            nb_params-=torch.sum((param == 0).int()).data[0]


def main(dataset,model_type):
    """[summary]

    Args:
        dataset ([string]): "cifar10" ou "cifar100"
        model_type ([string]): "models_distilled", "models_quantized", "models_pruned" ou "models_regularized"
    """
    listed_dir = f"./{dataset}/models/{model_type}"

    if dataset == "cifar10":
        ref_params = 5586981
        ref_flops  = 834362880
        for model in os.listdir(listed_dir):
            logging.info(f"profiling model in {model}")
            model, hparams = load_model_and_hyperparameters(model, listed_dir, int(dataset[dataset.find("1"):]))

            flops = profile(model, (1,3,32,32))
            flops = flops.item()

            params=model.numel()
            for param in model.parameters():
                # compte le nombre de 0, somme le total de 0 et on l'enlève du nombre de paramètres total
                if param is not None :
                    params-=torch.sum((param == 0).int()).data[0]
            
            if model_type == "models_quantized" or model_type == "models_distilled":
                type_params = type(params)
                params = params / (int(type_params[type_params.find("t")+1:])/hparams.nb_bits) # "t" de "Float"

            score_flops = flops / ref_flops
            score_params = params / ref_params
            score = score_flops + score_params

            ######### Il faut qu'on trouve un moyen d'enregister ça quelque part proprement #########
            print("Flops: {}, Params: {}".format(flops,params))
            print("Score flops: {} Score Params: {}".format(score_flops,score_params))
            print("Final score: {}".format(score))