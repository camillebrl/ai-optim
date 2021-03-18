import os
import csv
import pandas as pd
import torch
import torch.nn as nn
import logging
from pipeline import load_model_and_hyperparameters
import constants as CN

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
    print("Conv2d: S_c={}, F_in={}, F_out={}, P={}, operations={}".format(sh,fin,fout,x.size()[2:].numel(),int(total_ops)))
    # incase same conv is used multiple times
    m.total_ops += torch.Tensor([int(total_ops)])


def count_bn2d(m, x, y):
    x = x[0] # remove tuple
    nelements = x.numel()
    total_sub = 2*nelements
    total_div = nelements
    total_ops = total_sub + total_div
    m.total_ops += torch.Tensor([int(total_ops)])
    print("Batch norm: F_in={} P={}, operations={}".format(x.size(1),x.size()[2:].numel(),int(total_ops)))


def count_relu(m, x, y):
    x = x[0]
    nelements = x.numel()
    total_ops = nelements
    m.total_ops += torch.Tensor([int(total_ops)])
    print("ReLU: F_in={} P={}, operations={}".format(x.size(1),x.size()[2:].numel(),0,int(total_ops)))



def count_avgpool(m, x, y):
    x = x[0]
    total_add = torch.prod(torch.Tensor([m.kernel_size])) - 1
    total_div = 1
    kernel_ops = total_add + total_div
    num_elements = y.numel()
    total_ops = kernel_ops * num_elements

    m.total_ops += torch.Tensor([int(total_ops)])
    print("AvgPool: S={}, F_in={}, P={}, operations={}".format(m.kernel_size,x.size(1),x.size()[2:].numel(),0,int(total_ops)))

def count_linear(m, x, y):
    total_mul = m.in_features/2
    total_add = m.in_features - 1
    num_elements = y.numel()
    total_ops = (total_mul + total_add) * num_elements
    print("Linear: F_in={}, F_out={}, operations={}".format(m.in_features,m.out_features,int(total_ops)))
    m.total_ops += torch.Tensor([int(total_ops)])

def count_sequential(m, x, y):
    print ("Sequential: No additional parameters  / op")


def profile(model, input_size, custom_ops = {}):
    model.eval()
    def add_hooks(m):
        if len(list(m.children())) > 0: return
        m.register_buffer('total_ops', torch.zeros(1))

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

    x = torch.zeros(input_size,device=CN.DEVICE)
    model(x)

    total_ops = 0
    for m in model.modules():
        for m_2 in m.modules():
            if len(list(m_2.children())) > 0: continue
            total_ops += m_2.total_ops

    return total_ops



def scoring(dataset,model_type):
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

            params=0
            for param_tensor in model.parameters():
                if param_tensor is not None:
                    params+=len(torch.nonzero(param_tensor)) # tensor.nonzero donne les index des paramètres non nuls
                precision_origin=torch.finfo(param_tensor.dtype).bits
            if model_type == "models_quantized" or model_type == "models_distilled":
                params = params / (precision_origin / hparams.nb_bits)

            score_flops = flops / ref_flops
            score_params = params / ref_params
            score = score_flops + score_params

            with open("./scoring.csv",mode="a") as scoring_file:
                scoring_writer = csv.writer(scoring_file,delimiter=",",quotechar='"')
                scoring_writer.writerow(model_type+model,flops,params,score_flops,score_params,score)

scoring("cifar10","models_regularized")