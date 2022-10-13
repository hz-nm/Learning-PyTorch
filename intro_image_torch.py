"""
    This document will provide an introduction to using images with PyTorch.
    We will use a ready to download open access datasets from PyTorch/ TorchVision
    We will also transform the images for consumption by the model.
    Also defined is how to use DataLoader to feed batches of data to your model.
    Link -> https://pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
"""

import enum
from multiprocessing.dummy import freeze_support
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np


# ? Here we specify two transformations for our input images.
# ? transforms.ToTensor() -> converts images loaded by Pillow into PyTorch Tensors.
# ? transforms.Normalize() -> adjusts the values of the tensors so that their average is zero and their standard deviation is 0.5.
# ! Most activation functions have their strongest gradients around 0. So best to keep the values centered around there to speed up learning.


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# ! There are many more transformations available such as cropping, centering, rotation and reflection.

# * Create an instance of CIFAR-10 Dataset. 32x32 Image tiles representing 10 Classes of Objects. 6 of animals and 4 of Vehicles.
trainset = torchvision.datasets.CIFAR10(root='./data',          # specify a path 
                                         train=True,            # whether this dataset is being used for training or not.
                                         download=True,         # download the dataset if we are using online.
                                         transform=transform)   # The transformations we want to apply to the data.

# ? Above is an example of creating a dataset object in PyTorch. Dataset classes in PyTorch include downloadable datasets and utility dataset classes such as torchvision.datasets.ImageFolder
# ? which will read a folder of labeled images. 

# ! Once our dataset is ready we can give it to the DataLoader
trainloader = torch.utils.data.DataLoader(trainset,             # load the dataset
                                            batch_size=4,       # give us batches of 4 images from trainset
                                            shuffle=True,       # randomize their order
                                            num_workers=2       # spin up two workers to load data from disk.
                                            )

# ? Let's visualize the batches our DataLoader serves.
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5         # un-normalize the image.
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ! let's now train the model.
# ! -------------------------------
# * Get the testset

testset = torchvision.datasets.CIFAR10(root='./data',
                                        train=False,
                                        download=True,
                                        transform=transform)

testloader = torch.utils.data.DataLoader(testset,
                                            batch_size=4,
                                            shuffle=False,
                                            num_workers=2
                                            )

# ! NOW THE MODEL
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # 3 input channels, 6 output channels, 5x5 Convolution Kernel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    

# ? THE TRAINING LOOP
def training(net):

    # ? The last INGREDIENTSSS: Loss function and an optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # ! net.parameters() -> collection of all the learning weights in the model which is what the optimizer adjusts

    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward and backward and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:
                # print every 2000 mini batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/2000))
                running_loss = 0.0

    print('Finished training')



if __name__ == '__main__':
    freeze_support()
    
    net = Net()

    training(net)

    # ! Let's see how it does
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

