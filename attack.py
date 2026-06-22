import numpy as np
import json

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def fgsm_attack(model, loss, images, labels, eps):
    
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad = True
    
    outputs = model(images)
    
    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()
    
    attack_images = images + eps*images.grad.sign()
    attack_images = torch.clamp(attack_images, 0, 1)
    
    return attack_images

def pgd_attack(model, images, labels, eps, alpha, iters):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()
    
    ori_images = images.data
    
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        
        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min = 0, max =1).detach_()
        
    return images 

def generate_adversial_batch(model, images,labeles, attack_method,eps, alpha=None,iters=None):
    
    model.eval()
    model.zero_grad()
     
    if attack_method == 'fgsm':
        images.requires_grad = True
        
        outputs = model(images)
        loss = nn.CrossEntropyLoss()
        loss.backward()
        
        return fgsm_attack(model, loss, images, labeles, eps)
    
    elif attack_method == 'pgd':
        return pgd_attack(model, images, labeles, eps, alpha, iters)
    else:
        raise ValueError("Invalid attack method. Choose 'fgsm' or 'pgd'.")
        
        
