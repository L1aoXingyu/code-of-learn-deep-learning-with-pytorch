import torch
import torch.nn as nn
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable

# Hyper Parameters
num_epochs = 20
batch_size = 128
learning_rate = 1e-3

img_transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5], [0.5])])
# MNIST Dataset
train_dataset = dsets.MNIST(
    root='./data/', train=True, transform=img_transforms, download=True)

test_dataset = dsets.MNIST(
    root='./data/', train=False, transform=img_transforms)

# Data Loader (Input Pipeline)
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False)


# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),  # b, 16, 26, 26
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),  # b, 32, 24, 24
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #b, 32, 12, 12
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),  #b, 64, 10, 10
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3),  #b, 128, 8, 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  #b, 128, 4, 4
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


cnn = CNN()
if torch.cuda.is_available():
    cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        else:
            images = Variable(images)
            labels = Variable(labels)
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        _, correct_label = torch.max(outputs, 1)
        correct_num = (correct_label == labels).sum()
        acc = correct_num.data[0] / labels.size(0)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f, Acc: %.4f' %
                  (epoch + 1, num_epochs, i + 1,
                   len(train_dataset) // batch_size, loss.data[0], acc))

# Test the Model
cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %.4f ' %
      (correct / total))

# Save the Trained Model
torch.save(cnn.state_dict(), 'cnn.pth')