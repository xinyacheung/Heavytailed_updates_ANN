import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from update_tracker_2linear import MLPUpdateTracker
import pandas as pd
import wandb

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
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

def load_mnist_data(bs, selected_digits=None, train=True):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    
    torch.manual_seed(1234567)
    if train:
        dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    else:
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if selected_digits is not None:
        dataset.data = dataset.data[np.isin(dataset.targets.numpy(), selected_digits)]
        dataset.targets = dataset.targets[np.isin(dataset.targets.numpy(), selected_digits)]
    
    if train:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=250, shuffle=False)

    if train:
        print(f"Loaded {len(dataset)} training images")
    else:
        print(f"Loaded {len(dataset)} test images")

    return data_loader


def train_model(model, trainloader, testloader, criterion, optimizer, num_epochs=5, save_path=None, selected_digits=None, learning_rate=0.001, device=None, opti_suffix=None):

    tracker = MLPUpdateTracker(model)
    test_measure_list = []
    trainingloss=[]

    steps = 0
    model.train()
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(trainloader):
            inputs = inputs.view(inputs.size(0), -1).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            tracker.track_update_magnitude(steps)
            steps+=1
            wandb.log({"train_loss": loss.item()})
            trainingloss.append(loss.item())

            if steps % 100 == 0:
                test_measure_list.append(measure_model(model, testloader, device=device))
                print(f'[Epoch {epoch + 1}, Batch {i + 1}], test_acc = {test_measure_list[-1]:.2f}')
                model.train()
                print(f'Finished batch {i+1}')
            
            if steps>1 and steps % 5000 == 0:
                tracker.save_detailed_updates('../'+f'optim_{opti_suffix}_lr{learning_rate}_steps{steps}')
                tracker = MLPUpdateTracker(model)

    tracker.save_detailed_updates('../'+f'optim_{opti_suffix}_lr{learning_rate}_final')
    np.save('../'+f'trainingloss_optim_{opti_suffix}_lr{learning_rate}.npy', trainingloss)

def compute_loss_next(model, data_loader, criterion, device=None):
    model.eval()
    inputs, labels = next(iter(data_loader))
    inputs, labels = inputs.to(device), labels.to(device)
    inputs = inputs.view(inputs.size(0), -1)
    with torch.no_grad():
        outputs = model(inputs)
        return criterion(outputs, labels)

def compute_loss(model, data_loader, criterion, device=None):
    model.eval() 
    total_loss = 0  
    total_samples = 0  

    subset_size = 500

    with torch.no_grad(): 
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate loss and sample count
            batch_size = inputs.size(0)
            total_loss += loss * batch_size
            total_samples += batch_size
            
            if total_samples >= subset_size:
                break
    avg_loss = total_loss / total_samples
    return np.log10(avg_loss)

def measure_model(model, data_loader, measure='accuracy', device=None):
    model.eval() 
    correct = 0
    total = 0

    with torch.no_grad():  
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.view(inputs.size(0), -1) 
            outputs = model(inputs)  
            _, predicted = outputs.max(1) 
            total += labels.size(0)  
            correct += (predicted == labels).sum()
    accuracy = correct / total  
    return accuracy.item()

def main(selected_digits=None, num_epochs=100, learning_rate=0.001, bs=1, device=None, save_path=None, opti_suffix=None):
    trainloader = load_mnist_data(bs, selected_digits, train=True)
    testloader = load_mnist_data(bs, selected_digits, train=False)
    
    model = MLP(28*28, 128, output_size=len(selected_digits)).to(device)
    criterion = nn.CrossEntropyLoss()

    if opti_suffix == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    print(f"Training model, num_epochs={num_epochs}, learning_rate={learning_rate}")
    train_model(model, trainloader, testloader, criterion, optimizer, num_epochs, save_path, selected_digits, learning_rate, device, opti_suffix)


if __name__ == "__main__":
    select_list = [ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] ]
    save_path = 'your path here'  # Specify your save path here
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")

    num_epochs = 50
    bs=64

    for opti_suffix in ['SGD']: 
        for selected_digits in select_list:
            for lr in [0.01,0.1]:
                
                wandb.init(project="Criticality-MLP", group="MLP",
                        config={"dataset": "MNIST",
                                "selected_digits": select_list, "num_epochs": num_epochs, 
                                "architecture": "2 Layer MLP",
                                "optimizer": opti_suffix,
                                "learning_rate": lr, "device": device,
                                "batch_size": bs})
                
                main(selected_digits=selected_digits, num_epochs=num_epochs, 
                    learning_rate=lr, bs=bs, device=device, save_path=save_path, opti_suffix=opti_suffix)
                
                wandb.finish()