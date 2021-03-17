import torch
import torch.nn as nn
import numpy as np
import random
import logging
import tqdm

def hook(module, input, output):
    if hasattr(module, "_input_hook"):
        # input contient plusieurs inputs différents?
        # ça ne fonctionne sans [0] car ça renvoie un tuple...
        module._input_hook = input[0]
        module._output_hook = output
    else:
        setattr(module, "_input_hook", input[0])

class Distillation():
    def __init__(self, model_student, model_teacher, device):
        self.model_student = model_student
        self.model_teacher = model_teacher
        self.device = device

    def distillation_hilton(self, trainloader, regul_coef):
        regul = 0
        for m in self.model.modules():
            m.register_forward_hook(hook)
        n=64
        logging.info(f"Distillation on {len(self.model.modules())} modules with {n} batches")
        for mod, m in tqdm(enumerate(self.model.modules())):
            n = 64
            # récupère au hasard les indices de n batch dans trainloader
            subset_indices = [random.randint(0, len(trainloader) - 1) for _
                                in range(n)]
            output_student_batch=torch.Tensor(device="cpu")
            output_teacher_batch=torch.Tensor(device="cpu")
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if i in subset_indices:
                    torch.cat(self.model_student(inputs).to("cpu"),output_student_batch) # ce n'est pas cat, mais c'est quoi?
                    torch.cat(self.model_teacher(inputs).to("cpu"),output_teacher_batch) # ce n'est pas cat, mais c'est quoi?
            regul += regul_coef*torch.sum(output_teacher_batch*torch.log(output_teacher_batch/output_student_batch))
        return regul
