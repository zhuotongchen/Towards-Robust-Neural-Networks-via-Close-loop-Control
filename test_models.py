import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from Adversarial_attack import fgsm, random, pgd, CW_attack, Manifold_attack

def testing(test_loader, model, step_size, eps, attack='None', device=None):    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if attack == 'None':
            images_ = inputs
        elif attack == 'fgsm':
            images_ = fgsm(inputs, labels, eps, criterion, model)
        elif attack == 'random':
            images_ = random(inputs, labels, eps, criterion, model)
        elif attack == 'pgd':
            images_ = pgd(model, inputs, labels, criterion, num_steps=20, step_size=step_size, eps=eps)
        elif attack == 'cw':
            print('Processing CW attack on batch:', i)
            CW = CW_attack(model)
            images_ = CW.attack(inputs, labels, eps)
        outputs = model(images_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    return accuracy

def manifold_attack(test_loader, model, eps, basis, device):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        print('Processing CW attack on batch:', i)
        Man_attack = Manifold_attack(model, basis)
        images_ = Man_attack.attack(inputs, labels, eps)
        outputs = model(images_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    return accuracy        
    

def testing_save(test_loader, model, step_size, eps, attack, device, embedding_basis=None):    
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    
    images_adv_collection = []
    
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if attack == 'None':
            images_ = inputs
        elif attack == 'fgsm':
            images_ = fgsm(inputs, labels, eps, criterion, model)
        elif attack == 'random':
            images_ = random(inputs, labels, eps, criterion, model)
        elif attack == 'pgd':
            images_ = pgd(model, inputs, labels, criterion, num_steps=20, step_size=step_size, eps=eps)
        elif attack == 'cw':
            print('Processing CW attack on batch:', i)
            CW = CW_attack(model)
            images_ = CW.attack(inputs, labels, eps)
        elif attack == 'manifold':
            images_ = manifold_attack(inputs, labels, eps, criterion, model, embedding_basis)
        outputs = model(images_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        images_adv_collection.extend(images_)
        
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    dataset = CustomDataset(data_set=images_adv_collection, label_set=test_loader.dataset.targets)
    data_loader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    torch.save(data_loader, 'cw_attack_eps_{}'.format(eps))
    return accuracy

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_set, label_set):
        self.dataset = data_set
        self.targets = label_set

    def __len__(self):
        return len(self.targets)
   
    def __getitem__(self, index):
       return self.dataset[index], self.targets[index]
   
    