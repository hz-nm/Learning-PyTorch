"""
The following tutorial is available on,
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/74249e7f9f1f398f57ccd094a4f3021b/transfer_learning_tutorial.ipynb

In this tutorial, we will learn how to train a convolutional neural network for image classification using Transfer Learning.
In Practice, very few people train an entire Convolutional Network from scratch with Random Initialization.
It is common to pretrain a ConvNet on a very large Dataset and then use it as an initialization or a fixed feature extractor.

[][][][][][]
Finetuning the ConvNet: 
            Instead of Random Initialization, we initialize the network with a pretrained Network. Rest of the training looks as usual.
ConvNet as a fixed feature extractor:
            Here we will freeze the weights for all of the network except that of the final fully connected layer.
            The final layer is replaced with a new one with Random Weights and only this layer is trained.
"""
# %%
from __future__ import print_function, division
from multiprocessing.dummy import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy



# %%
# ! Training the Model
# * We will schedule the learning rate and save the best model.
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # set model to training mode.
            else:
                model.eval()    # set the model to evaluate mode.

            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # ! forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # ! backward + optimize only if in training phase.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load the best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
# ! Generic Function to Visualize Model Predictions
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %%
if __name__ == '__main__':
    cudnn.benchmark = True
    plt.ion()       # ! interactive mode??

    # %
    # ! Load the Data
    # ? We will use torchvision and torch.utils.data packages for loading the data.
    # ? We will train a model to classify bees and ants. (120 Images for each for training 75 for validation)
    # ? Even with a small dataset, transfer learning will work well and the model will generalize reasonably well.
    # * Data Augmentation and Normalization for training
    # * Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # %
    data_dir = './hymenoptera_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                        for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
                        for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}       # basically dictionaries in which two repeated codes are jammed into one.
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ! let's visualize a few images
    def imshow(input_image, title=None):
        # ? IMSHOW for tensor
        inp = input_image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)        # pause a bit so that plots are updated.

    # %
    # * Get a batch of training data.
    inputs, classes = next(iter(dataloaders['train']))
    # * Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])


    # freeze_support()
    # ! Let's now look at finetuning our CONVNET
    # * Load a pretrained model and reset final fully connected layer.
    model_ft = models.resnet18(pretrained=True) # a fully trained model
    num_ftrs = model_ft.fc.in_features          # number of features

    # ? Here the size of each output sample is set to 2.
    # ? Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)        # can also maybe put hidden layer in between. Like in pytorch_object_detection.py
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # * Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # * Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # 
    # ! Finally we can train and evaluate.
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25)
    visualize_model(model_ft)

    # 
    # ! \/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/\
    # ! Convnet as a FIXED Feature Extractor
    # * Here we need to freeze all the network except for the final layer.
    # * We need to set required_grad=False to freeze the parameters so that the Gradients are not computed in backward().
    model_conv = torchvision.models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False

    # * Parameters of newly constructed modules have requires_grad = True by default.
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)
    model_conv = model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    # * See that only parameters of final layer are being optimized as opposed to before.
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
    # * Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler =  lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # 
    # ! Let's now train and Evaluate this model
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)

    visualize_model(model_conv)

    plt.ioff()
    plt.show()

    

# %%
