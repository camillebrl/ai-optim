import torch
import torch.nn as nn
import numpy as np
import random
import logging
import tqdm

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

    def distillation_hilton(self, trainloader, regul_coef):
        regul = 0
        for m_student, m_teacher in zip(self.model_student.modules(),self.model_teacher.modules()):
            m_student.register_forward_hook(hook)
            m_teacher.register_forward_hook(hook)
        n=64
        logging.info(f"Distillation on {len(self.model.modules())} modules with {n} batches")
        for m_student, m_teacher in zip(self.model_student.modules(),self.model_teacher.modules()):
            # récupère au hasard les indices de n batch dans trainloader
            subset_indices = [random.randint(0, len(trainloader) - 1) for _
                                in range(n)]
            output_student_batch=torch.Tensor(device="cpu")
            output_teacher_batch=torch.Tensor(device="cpu")
            for i, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                if i in subset_indices:
                    torch.cat(m_student._output_hook.to("cpu"),output_student_batch) # ce n'est pas cat, mais c'est quoi?
                    torch.cat(m_teacher._output_hook.to("cpu"),output_teacher_batch) # ce n'est pas cat, mais c'est quoi?
            regul += regul_coef*torch.sum(output_teacher_batch*torch.log(output_teacher_batch/output_student_batch))
        return regul