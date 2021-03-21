import torch
import torch.nn as nn
import numpy as np
import random


class Orthogo():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.target_modules = []
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                self.target_modules.append(m)

    def soft_orthogonality_regularization(self, reg_coef):
        regul = 0
        for i, m in enumerate(self.target_modules):
            width = np.prod(list(m.weight.data[0].size()))
            height = m.weight.data.size()[0]
            w = m.weight.data.view(width, height)
            reg_coef_i = reg_coef
            regul += reg_coef_i * torch.norm(
                torch.transpose(w, 0, 1).matmul(w) - torch.eye(height,
                                                               device=self.device)) ** 2
            # reg_grad=4*reg_coef*w*(torch.transpose(w,0,1)*w-torch.eye(height))
        # le terme de régularisation est sur tous les modules! (somme)
        return regul

    def double_soft_orthogonality_regularization(self, reg_coef):
        regul = 0
        for m in self.target_modules:
            width = np.prod(list(m.weight.data[0].size()))
            height = m.weight.data.size()[0]
            w = m.weight.data.view(width, height)
            regul += reg_coef * (torch.norm(
                torch.transpose(w, 0, 1).matmul(w) - torch.eye(height,
                                                               device=self.device)) ** 2 + torch.norm(
                w.matmul(torch.transpose(w, 0, 1)) - torch.eye(height,
                                                               device=self.device)) ** 2)
        return regul

    def mutual_coherence_orthogonality_regularization(self, reg_coef):
        regul = 0
        for m in self.target_modules:
            width = np.prod(list(m.weight.data[0].size()))
            height = m.weight.data.size()[0]
            w = m.weight.data.view(width, height)
            regul += reg_coef * (torch.max(torch.abs((torch.transpose(w, 0,
                                                                      1).matmul(
                w) - torch.eye(height, device=self.device)) ** 2)))
        return regul

    def spectral_isometry_orthogonality_regularization(self, reg_coef):
        regul = 0
        for m in self.target_modules:
            width = np.prod(list(m.weight.data[0].size()))
            height = m.weight.data.size()[0]
            w = m.weight.data.view(width, height)
            v = torch.rand(w.size()[1],w.size()[0],device=self.device)
            for _ in range(2):
                u = (torch.transpose(w,0,1).matmul(w) - torch.eye(height,device=self.device)).matmul(v)
                v = (torch.transpose(w,0,1).matmul(w) - torch.eye(height,device=self.device)).matmul(u)
            regul += reg_coef * ( torch.norm(v) / torch.norm(u) )
        return regul