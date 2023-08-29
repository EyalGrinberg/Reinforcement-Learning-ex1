import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from matplotlib import pyplot as plt

# Hyper Parameters
input_size = 784
num_classes = 10
num_epochs = 100
batch_size = 100
learning_rate = 1e-3
moment = 0.9
hidden_size = 500

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model
class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.fc3(self.fc2(self.fc1(x)))
        return out

net = Net(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum = moment)

# Train the Model
loss_over_time = []
for epoch in range(num_epochs):
    loss_over_time.append(0)
    for i, (images, labels) in enumerate(train_loader):
        # Convert torch tensor to Variable
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        # Forward + Backward + Optimize
        pred = net(images)
        loss = criterion(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_over_time[epoch] += loss.item()
    loss_over_time[epoch] /= len(train_loader)

# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    correct += torch.sum(torch.argmax(net(images), axis = 1) == labels)
    total += labels.size(0)

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
plt.plot(range(1,101), loss_over_time)
plt.title("Model 3 Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.show()
# Save the Model
torch.save(net.state_dict(), 'model3.pkl')