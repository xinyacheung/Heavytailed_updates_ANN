import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from update_tracker_cnn import CNN10UpdateTracker
from torch.utils.data import DataLoader
import pandas as pd
import wandb


class CNN10(nn.Module):
    def __init__(self):
        super(CNN10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # CIFAR shape 32x32，after pooling 8x8
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def get_weights_as_vector(self):
        weights = []
        for param in self.parameters():
            weights.append(param.data.view(-1))
        return torch.cat(weights)

    def set_weights_from_vector(self, weight_vector):
        start = 0
        for param in self.parameters():
            param_length = param.numel()
            param.data = weight_vector[start:start + param_length].view(param.shape)
            start += param_length

def load_cifar_data(name=None, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    torch.manual_seed(1234567)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(testset, batch_size=250, shuffle=False) 

    if name == 'cifar10':
        print(f'Load CIFAR-10 dataset, trainset size: {len(trainset)}, testset size: {len(testset)}')
    elif name == 'cifar100':
        print(f'Load CIFAR-100 dataset, trainset size: {len(trainset)}, testset size: {len(testset)}')
    
    return trainloader, testloader

def train_model(model, trainloader, testloader, criterion, optimizer, bs, num_epochs=5, save_path=None, name=None, learning_rate=0.001, device=None, opti_suffix=None):
    
    tracker = CNN10UpdateTracker(model)
    model.train()

    steps = 0
    test_measure_list = []
    trainingloss=[]

    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.view(-1, 3, 32, 32).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            ### if track the updates
            tracker.track_update_magnitude(steps)
            
            steps+=1

            wandb.log({"train_loss": loss.item()})
            trainingloss.append(loss.item())

            if steps % 100 == 0:
                test_measure_list.append(measure_model(model, testloader, device=device))
                print(f'[Epoch {epoch + 1}, Batch {i + 1}], test_acc = {test_measure_list[-1]:.2f}')
                print(f'Finished batch {i+1}')

            # save the tracker steply to avoid memory overflow
            if steps>1 and steps % 200 == 0:
                tracker.save_detailed_updates('../'+f'optim_{opti_suffix}_lr{learning_rate}_steps{steps}')
                tracker = CNN10UpdateTracker(model)

    tracker.save_detailed_updates('../'+f'optim_{opti_suffix}_lr{learning_rate}_final')
    np.save('../'+f'trainingloss_optim_{opti_suffix}_lr{learning_rate}.npy', trainingloss)

def compute_loss(model, data_loader, criterion, device):
    model.eval() 
    total_loss = 0  
    total_samples = 0  

    subset_size = 500

    with torch.no_grad(): 
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 3, 32, 32)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate loss and sample count
            batch_size = inputs.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size
            
            if total_samples >= subset_size:
                break
    avg_loss = total_loss / total_samples
    return avg_loss

def measure_model(model, data_loader, measure='accuracy', device=None):
    model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(-1, 3, 32, 32).to(device)
            outputs = model(inputs)  
            _, predicted = outputs.max(1) 
            total += labels.size(0)  
            correct += (predicted == labels).sum()
    accuracy = correct / total  
    return accuracy.item()


def main(name=None, bs=None, num_epochs=10, learning_rate=0.001, save_path=None, device=None, opti_suffix=None):
    trainloader, testloader = load_cifar_data(name=name, batch_size=bs)  
    if name == 'cifar10':
        model = CNN10().to(device)
    criterion = nn.CrossEntropyLoss()

    if opti_suffix == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Training model, num_epochs={num_epochs}, learning_rate={learning_rate}")
    train_model(model, trainloader, testloader, criterion, optimizer, bs, num_epochs, save_path, name, learning_rate, device, opti_suffix)
    

if __name__ == "__main__":
    select_list = ['cifar10']
    save_path = 'your path here'  # Specify your save path here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    num_epochs = 50
    bs=64

    for opti_suffix in ['SGD']:
        for selected_name in select_list:
            for lr in [0.01]:

                wandb.init(project="Criticality-CNN", group=f"CNN",
                config={"dataset": selected_name,
                        "num_epochs": num_epochs, 
                        "architecture": "CNNs",
                        "optimizer": opti_suffix,
                        "learning_rate": lr, "device": device,
                        "batch_size": bs,
                        "note":"add batch normalization"})
                
                main(name=selected_name, bs=bs, num_epochs=num_epochs, 
                    learning_rate=lr, save_path=save_path, device=device, opti_suffix=opti_suffix)
                
                wandb.finish()