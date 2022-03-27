import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.RandomRotation(30)
                               ])

# Download and load the training data
trainset = datasets.FashionMNIST('MNIST_data/', download = True, train = True, transform = transform)
testset = datasets.FashionMNIST('MNIST_data/', download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 64, shuffle = True)
testloader = torch.utils.data.DataLoader(testset, batch_size = 64, shuffle = True)

device = torch.device("cuda:0")



class ConvNN(nn.Module):
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)          #32x14x14
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)        #64x7x7
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)              #128x7x7
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, 7)                        #256x1x1
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
        x = F.leaky_relu(self.maxpool(self.bn1(self.conv1_2(x))), negative_slope=0.01)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
        x = F.leaky_relu(self.maxpool(self.bn2(self.conv2_2(x))), negative_slope=0.01)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.01)
        x = F.leaky_relu(self.bn3(self.conv3_2(x)), negative_slope=0.01)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.01)
        x = F.leaky_relu(self.bn4(self.conv4_2(x)), negative_slope=0.01)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        x = self.fc2(x)

        return x

#model = Network()
model = ConvNN()

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.SGD(model.parameters(), lr = 0.0014)

epochs = 35

train_losses, test_losses = [], []

for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:

        images, labels = images.to(device), labels.to(device)
        # Training pass
        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        test_loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computation
        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()

            # Validation pass
            for images, labels in testloader:

                images, labels = images.to(device), labels.to(device)

                log_ps = model(images)
                test_loss += criterion(log_ps, labels)

                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))

        print("Epoch: {}/{}..".format(e + 1, epochs),
              "Training loss: {:.3f}..".format(running_loss / len(trainloader)),
              "Test loss: {:.3f}..".format(test_loss / len(testloader)),
              "Test Accuracy: {:.3f}".format(accuracy / len(testloader)))


