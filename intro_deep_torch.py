
# ! Expressing LeNET Deep Learning model using PyTorch
import torch
import torch.nn as nn                   # parent object for PyTorch models
import torch.nn.functional as F         # for the activation function

class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # ? 1 input image channel (black and white), 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an Affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*6*6, 120)   # 6x6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max Pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]     # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
# * The above code demonstrates a typical PyTorch model.
# * It inherits from the torch.nn.Module. Modules may be nested.
# * A model will have __init__() class from which it instantiates its layers and loads any data artifacts it may need. For e.g an NLP Model may inherit a vocabulary.
# * A Model will have a forward() function. This is where the actual computation happens. An input is passed through the network layers and various functions to generate an output.
# * Other than that we can build out our model class like any other Python class. Adding whatever properties and methods you need to support your model's computation.

# ! Let's instantiate this object and run a sample input through it.
net = LeNet()       # instantiate the LeNet class and print the net object. 
print(net)          # this will report layers it has created and their shapes and parameters. This can provide a handy overview of the model.

input = torch.rand(1, 1, 32, 32)        # stand in for a 32x32 black and white image. Normally we would load an image tile and convert it into a tensor of this shape.
print('\nImage batch shape: {}'.format(input.shape)) # An extra dimension is added above. This is the batch dimension. A batch of 16 of our image tiles would have the shape (16, 1, 32, 32)
                                                     # Since we are only using one image we create a batch of 1 with shape (1, 1, 32, 32)


output = net(input)         # We ask the model for inference by calling it like a function on our input. The output of this call represents the models confidence.
print('\nRaw Output:')
print(output)
print(output.shape)         # The output also has a batch dimension which corresponds to the input batch dimension.