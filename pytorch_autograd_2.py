"""Autograd in Training
"""

import torch

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(1000, 1000)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

model = TinyModel()

# One thing you might notice is that we never specify requires_grad=True for the model’s layers. 
# Within a subclass of torch.nn.Module, it’s assumed that we want to track gradients on the layers’ weights for learning.

# If we look at the layers of the model, we can examine the values of the weights, and verify that no gradients have been computed yet

print(model.layer2.weight[0][0:10]) # just a small slice
print(model.layer2.weight.grad)

# Let’s see how this changes when we run through one training batch. 
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
prediction = model(some_input)

loss =  (ideal_output - prediction).pow(2).sum()
print(loss)

# Now let's call loss.backward() and see what happens
loss.backward()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])

# ! Let's run the optimizer since gradients had been computed in the previous step but the weights remain unchanged.
optimizer.step()
print(model.layer2.weight[0][0:10])
print(model.layer2.weight.grad[0][0:10])

# ! We can see layer 2's weights have now also changed.
# One important thing about the process: After calling optimizer.step(), 
# ! you need to call optimizer.zero_grad(), or else every time you run loss.backward(), 
# the gradients on the learning weights will accumulate:
print(model.layer2.weight[0][0:10])

for i in range(0, 5):
    prediction = model(some_input)
    loss = (ideal_output - prediction).pow(2).sum()
    loss.backward()

print(model.layer2.weight[0][0:10])

optimizer.zero_grad()

print(model.layer2.weight[0][0:10])

# After running the cell above, you should see that after running loss.backward() multiple times, the magnitudes of most of the gradients will be much larger. Failing to
# zero the gradients before running your next training batch will cause the gradients to blow up in this manner, causing incorrect and unpredictable learning results.


