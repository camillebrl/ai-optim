import torch
import torch.nn as nn
import numpy as np
import constants as CN
from sklearn.cluster import KMeans
from sklearn.utils.extmath import row_norms, squared_norm
from numpy.random import RandomState
import scipy as sc



def regul_deep_k_means(model, device, regul_coef):
    # https://arxiv.org/pdf/1806.09228.pdf 
    # min(E(W) +λ2[Tr(WTW)−Tr(FTWTWF)])
    target_modules = []
    for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                target_modules.append(m)
    regul = 0
    for i, m in enumerate(target_modules):
        w = m.weight.data.view(m.weight.data.size()[2],-1).to("cpu") # share s*N avec N=s*C*M
        a,b,F = torch.svd(w)
        regul += (regul_coef / 2) * (torch.trace(torch.transpose(w,0,1).matmul(w))) - torch.trace(torch.transpose(F,0,1).matmul(torch.transpose(w,0,1).matmul(w).matmul(F)))
    return regul


def cluster_base(nb_clusters, tensor, device):
    tensor_flat=tensor.view(tensor.size()[0],tensor.size()[1]*tensor.size()[2]*tensor.size()[3])
    tensor_flat=tensor_flat.to("cpu").detach()
    kmeans_result = KMeans(n_clusters=nb_clusters, init='k-means++', random_state = 1).fit(tensor_flat)
    labels = kmeans_result.labels_
    centers = kmeans_result.cluster_centers_
    weight_vector_compress = np.zeros((tensor_flat.shape[0], tensor_flat.shape[1]), dtype=np.float32)
    for v in range(tensor_flat.shape[0]):
        weight_vector_compress[v, :] = centers[labels[v], :]
    results=torch.Tensor(weight_vector_compress,device="cpu")
    results=results.to(device)
    return results.view(tensor.size())


class Cluster():
    def __init__(self, model, nb_clusters, device):

        self.model = model
        self.nb_clusters = nb_clusters
        self.device = device
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                count_targets = count_targets + 1
        start_range = 0
        end_range = count_targets - 1
        self.bin_range = np.linspace(start_range,
                                     end_range, 
                                     end_range - start_range + 1) \
                                    .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.target_modules = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

    def clustering(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(
                cluster_base(self.nb_clusters,
                            self.target_modules[index].data,
                            self.device))
    

                    