
# ! This folder should work as an introduction to PyTorch
# ! environment for this -> object_yolo

# ? This tutorial introduces you to a complete ML Workflow implemented in PyTorch.
# ? We will use the FashionMNIST dataset to train a neural network that predicts if an input image belongs to one of the following classes,
# ? T-Shirt/Top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag or Ankle boot

# * PyTorch has two primitives to work with data:
# *     torch.utils.data.DataLoader :   Dataset stores the samples and their corresponding labels.
# *     torch.utils.data.Dataset :  DataLoader wraps an iterable around the Dataset.

import enum
from os import cpu_count
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# * PyTorch offers domain specific libraries such as TorchText, TorchVision and TorchAudio. All of them include datasets.
# * For this tutorial we will be using a TorchVision dataset.
# ? TORCHVISION.DATASETS all datasets -> https://pytorch.org/vision/stable/datasets.html

# * For this tutorial, we use the FashionMNIST dataset.
# * Every TorchVision Dataset includes two Arguments
# * transform and target_transform to modify the samples and labels respectively.

# ! Download the training data from open datasets
training_data = datasets.FashionMNIST(
    root = "data",
    train = True,
    download = True,
    transform = ToTensor(),
)

# ! Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)


# ! We pass the dataset as an argument to DataLoader. This wraps an iterable over our dataset, and supports automatic batching, sampling and shuffling
# * Here we define a batch of 64, i.e. each element in the dataloader iterable will return a batch of 64 features and labels.
batch_size = 64

# * Create the DataLoaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f'Shape of X [N, C, H, W]: {X.shape}')
    print(f'Shape of y: {y.shape} {y.dtype}')
    break

# ! Let's now create some model. To create/define a neural network in PyTorch, we create a class that inherits from nn.Module. We define layers of the network
# ! in the __init__ function and specify how data will pass through the network in the forward function. To accelerate operations of a neural network, we move it to GPU

# Get cpu our gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            # nn.Linear(512, 512),        # Two numbers basically identify an input figure and an output figure.
            # nn.ReLU(),                  # These two additional layers didn't improve the performance but actually downgraded it.
            nn.Linear(512, 10)            # 10 classes
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# ! OPTIMIZING MODEL PARAMETERS
# * To train a model, we need a loss function and an optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# * In a single training loop the model makes predictions on the training dataset (fed into it in batches)
# * and backpropagates the prediction error to adjust the model's parameters.

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute the prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')


# ! We also check the Model's performance against the test dataset to ensure it is learning.

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss, correct = 0, 0
    with torch.no_grad():           # disable gradient calculcation for better inference result.
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# ! Define the epochs to train the model
epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)

print("DONE!")

# ! SAVE THE MODEL
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch model state to model.pth")


# ! Loading the model
model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))
# ? OUT: <ALL KEYS MATCHED SUCCESSFULLY>

# ! This model can now be used to make predictions
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()

x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: '{predicted}', Actual: '{actual}'")




