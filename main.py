import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

from model import *
from train_models import training
from test_models import testing, manifold_attack, testing_save

device = 'cuda' if torch.cuda.is_available() else 'cpu'


workers = 0
epochs = 300
start_epoch = 0
batch_size = 128
learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
print_freq = 50

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=batch_size, shuffle=True,
    num_workers=workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=128, shuffle=False,
    num_workers=workers, pin_memory=True)

_lambda = [0.5, 0.5]
eps_defense = 4
eps_attack = 8
num_classes = 10
training_method = 'pgd'

# def main():
#     print('==> Building model..')
#     net = resnet20()
#     net = net.to(device)
    # criterion = nn.CrossEntropyLoss().cuda()
    # training(train_loader, test_loader, net, epochs, start_epoch, learning_rate, momentum, weight_decay, num_classes, device,
    #           eps_defense=eps_defense, eps_attack=eps_attack, train_method=training_method)

# if __name__ == '__main__':
#     main()

net = resnet20()
net = net.to(device)
net.load_state_dict(torch.load('models/resnet20_model.ckpt', map_location=device))


# testing(test_loader, net, step_size=0.5, eps=2., attack='pgd', device=device)
# testing(test_loader, net, step_size=1., eps=4., attack='pgd', device=device)
# testing(test_loader, net, step_size=2., eps=8., attack='pgd', device=device)

from pca_source import Linear_Control

Lin = Linear_Control(net)
# # Lin.compute_Princ_basis(train_loader)
Lin.Princ_vec = torch.load('Principle_axis/Princ_axis')
Lin.from_basis_projection()

Lin.PMP_testing(test_loader)

# Lin.Princ_vec = torch.load('Principle_axis/Princ_vec')
