'''
proxy-synthesis
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.distributions.multivariate_normal import MultivariateNormal

def proxy_synthesis(input_l2, proxy_l2, target, ps_alpha, ps_mu):
    '''
    input_l2: [batch_size, dims] l2-normalized embedding features
    proxy_l2: [n_classes, dims] l2-normalized proxy parameters
    target: [batch_size] Note that adjacent labels should be different (e.g., [0,1,2,3,4,5,...])
    ps_alpha: alpha for beta distribution
    ps_mu: generation ratio (# of synthetics / batch_size)
    '''

    input_list = [input_l2]
    proxy_list = [proxy_l2]
    target_list = [target]

    ps_rate = np.random.beta(ps_alpha, ps_alpha)

    input_aug = ps_rate * input_l2 + (1.0 - ps_rate) * torch.roll(input_l2, 1, dims=0)
    proxy_aug = ps_rate * proxy_l2[target,:] + (1.0 - ps_rate) * torch.roll(proxy_l2[target,:], 1, dims=0)
    input_list.append(input_aug)
    proxy_list.append(proxy_aug)
    
    n_classes = proxy_l2.shape[0]
    # * ps_mu를 곱했다. 이래야 맞지 않냐?
    pseudo_target = torch.arange(n_classes, n_classes + input_l2.shape[0] * ps_mu, dtype=torch.long).cuda()
    target_list.append(pseudo_target)

    embed_size = int(input_l2.shape[0] * (1.0 + ps_mu))
    proxy_size = int(n_classes + input_l2.shape[0] * ps_mu)
    input_large = torch.cat(input_list, dim=0)[:embed_size,:]
    proxy_large = torch.cat(proxy_list, dim=0)[:proxy_size,:]
    target = torch.cat(target_list, dim=0)[:embed_size]
    
    input_l2 = F.normalize(input_large, p=2, dim=1)
    proxy_l2 = F.normalize(proxy_large, p=2, dim=1)

    return input_l2, proxy_l2, target




def svd_confidence_control_without_input_cat(input_l2, proxy_l2, target, cc_mu, ps_mu):
    input_list = []
    proxy_list = []
    target_list = []

    number_of_same_class_instance = torch.sum(target == target[0]).item()

    for i in range(int(target.shape[0] / number_of_same_class_instance)):
        with torch.no_grad():
            x = input_l2[
                i * number_of_same_class_instance: i * number_of_same_class_instance + number_of_same_class_instance]
            mean = torch.mean(x, dim=0)
            cov = torch.cov(x.t())

            distrib = MultivariateNormal(loc=mean, covariance_matrix=cov)
            sample = distrib.rsample(sample_shape=int(number_of_same_class_instance * cc_mu))
            input_list.append(torch.from_numpy(sample).to(proxy_l2.device))

            eigvec = torch.lobpcg(cov, k=1, method="ortho", niter=50)[1]
            proxy_size = torch.linalg.vector_norm(proxy_l2[target[i * number_of_same_class_instance].item()])
            eigvec = (eigvec / torch.linalg.vector_norm(eigvec)) * proxy_size
            proxy_list.append(eigvec.t())

            pseudo_target = torch.full(size=(int(number_of_same_class_instance * cc_mu),), fill_value=proxy_l2.shape[0] + input_l2.shape[0] * ps_mu + i, dtype=torch.long).to(proxy_l2.device)
            target_list.append(pseudo_target)


    input_l2 = F.normalize(torch.cat(input_list, dim=0), p=2, dim=1)
    proxy_l2 = F.normalize(torch.cat(proxy_list, dim=0), p=2, dim=1)

    return input_l2, proxy_l2, torch.cat(target_list, dim=0)

class Norm_SoftMax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=23.0, ps_mu=0.0, ps_alpha=0.0,cc_mu=1.0, normalize=True, confidence_control_mode="non"):
        super(Norm_SoftMax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        self.normalize = normalize
        self.confidence_control_mode = confidence_control_mode
        self.cc_mu = cc_mu
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1) if self.normalize else input
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1) if self.normalize else self.proxy


        if self.confidence_control_mode == "non":
            if self.ps_mu > 0.0:
                input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target, self.ps_alpha, self.ps_mu)
            sim_mat = input_l2.matmul(proxy_l2.t())
            logits = self.scale * sim_mat
            loss = F.cross_entropy(logits, target)
            return loss
        if self.confidence_control_mode == "naive":
            if self.ps_mu > 0.0:
                input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target, self.ps_alpha, self.ps_mu)
            sim_mat = input_l2.matmul(proxy_l2.t())
            sim_mat = torch.cat((sim_mat, sim_mat[tuple(range(target.shape[0])), target].reshape(-1, 1)), dim=1)
            logits = self.scale * sim_mat
            loss = F.cross_entropy(logits, target)
            return loss
        if self.confidence_control_mode == "svd":
            cc_input_l2, cc_proxy_l2, cc_target = svd_confidence_control_without_input_cat(input_l2, proxy_l2, target,
                                                                                           self.cc_mu, self.ps_mu)
            if self.ps_mu > 0.0:
                input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target, self.ps_alpha, self.ps_mu)

            input_l2 = torch.cat((input_l2, cc_input_l2), dim=0)
            proxy_l2 = torch.cat((proxy_l2, cc_proxy_l2), dim=0)
            target = torch.cat((target, cc_target), dim=0)

            sim_mat = input_l2.matmul(proxy_l2.t())
            logits = self.scale * sim_mat
            loss = F.cross_entropy(logits, target)
            return loss

class Proxy_NCA(nn.Module):
    def __init__(self, input_dim, n_classes, scale=10.0, ps_mu=0.0, ps_alpha=0.0):
        super(Proxy_NCA, self).__init__()
        self.scale = scale
        self.n_classes = n_classes
        self.ps_mu = ps_mu
        self.ps_alpha = ps_alpha
        self.proxy = Parameter(torch.Tensor(n_classes, input_dim))
        
        init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
    
    
    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)

        if self.ps_mu > 0.0:
            input_l2, proxy_l2, target = proxy_synthesis(input_l2, proxy_l2, target,
                                                         self.ps_alpha, self.ps_mu)
 
        dist_mat = torch.cdist(input_l2, proxy_l2) ** 2
        dist_mat *= self.scale
        pos_target = F.one_hot(target, dist_mat.shape[1]).float()
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1))

        return loss
