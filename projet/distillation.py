import torch
import torch.nn as nn
import numpy as np
import random
import logging
import tqdm
import torch.nn.functional as F

def hook(module, input, output):
    if hasattr(module, "_output_hook"):
        module._output_hook = output
    else:
        setattr(module, "_output_hook", output)

class Distillation():
    def __init__(self, model_student, model_teacher, device):
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.device = device

    def distillation_hilton(self, output_student,output_teacher, tau=2, l=1):
        regul = 0
        output_stud=output_student / tau 
        output_teach=output_teacher / tau 
        output_stud=F.softmax(output_stud,dim=1)
        output_teach=F.softmax(output_teach,dim=1)
        regul=-torch.sum(output_teach*torch.log(output_stud))*l
        return regul