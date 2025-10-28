import numpy
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Lambda, Compose

# IMPORT RAW IMAGES AND PRINT THE VALUES OF AT THE FIRST ONE
trainset = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)
print(trainset)

print(" ")
trainloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
trainloader
for inputs, labels in trainloader:
    print("Labels:", labels)
    # print("input shape", inputs.shape)
    # print("Inputs:", inputs)
    break  # This will break after printing the first batch.

inputs_np = inputs.detach().numpy()
inputs_np_squeezed = numpy.squeeze(inputs_np, axis=(0, 1))
print("image shape =", inputs_np_squeezed.shape)
print("image pixels =", inputs_np_squeezed)

# IMPORT + NORMALIZE RAW DATA, THEREBY CREATING IMAGES WITH NORMALIZED PIXEL VALUES
print(" ")
print(" ")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset_norm = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                               transform=transform)
trainloader_norm = torch.utils.data.DataLoader(trainset_norm, batch_size=1, shuffle=False)
for inputs_norm, labels_norm in trainloader_norm:
    print("Labels norm:", labels_norm)
    # print("Inputs trainloader_norm:", inputs2)
    break  # This will break after printing the first batch.

inputs_norm_np = inputs_norm.detach().numpy()
inputs_norm_np_squeezed = numpy.squeeze(inputs_norm_np, axis=(0, 1))
print("image norm shape =", inputs_norm_np_squeezed.shape)
print("image norm pixels =", inputs_norm_np_squeezed)

# First run the first code at "problem 2 to upload FashionMNist"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel, 32 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)    # Max pooling layer with 2x2 window to reduce spatial dimensions by half.
        self.conv2 = nn.Conv2d(32, 64, 3) # Conventional 32 input channels, 64 filters, 3x3 kernel
        self.fc1 = nn.Linear(64 * 5 * 5, 128) # Fully connected layer, input size = 64x5x5, output = 128
        self.fc2 = nn.Linear(128, 10)         # Fully connected layer, input = 128, output = 10 classes

    def forward(self, x): # Defines how the input passes through the network.
        x = self.pool(torch.relu(self.conv1(x))) #Apply the first convolutional layer to the input. Conv1 → ReLU → MaxPool
        x = self.pool(torch.relu(self.conv2(x))) # Conv2 → ReLU → MaxPool
        x = x.view(-1, 64 * 5 * 5)  # PyTorch’s equivalent to numpy.reshape() ## Flatten the 3D tensor to a 1D vector
        x = torch.relu(self.fc1(x)) # Fully connected layer 1 → ReLU
        x = self.fc2(x)  # Fully connected layer 2 (output logits) without softmax
        return x

# Initialize the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load and preprocess the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(1):  # Change the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the parameters of the trained model
torch.save(net.state_dict(), 'fashion_mnist_cnn.pth')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# Define the CNN model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)  # PyTorch’s equivalent to numpy.reshape()
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # No softmax at the end although classification
        return x

# Initialize the model
net = Net()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load and preprocess the FashionMNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(10):  # Change the number of epochs as needed
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# Save the parameters of the trained model
torch.save(net.state_dict(), 'fashion_mnist_cnn.pth')
loaded_model = Net() # initializa the model
loaded_model.load_state_dict(torch.load('fashion_mnist_cnn.pth'))
loaded_model.eval() # Put it in evaluation mode


# Load and preprocess the FashionMNIST TEST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Perform the testing batchwise
correct_test = 0
total_test = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = loaded_model(images)
        predicted_values, predicted_indices = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted_indices == labels).sum().item()


# Calculate Accuracy
accuracy_test = 100 * correct_test / total_test
print(f'Accuracy on the test dataset: {accuracy_test:.2f}%')
print("No of testexamples: ", total_test)
