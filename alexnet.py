import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()

        self.fc6 = nn.Linear(256 * 3 * 3, 1024)
        self.fc7 = nn.Linear(1024, 512)
        self.fc8 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.relu3(x)
        x = x.view(-1, 256 * 3 * 3)
        x = self.fc6(x)
        x = F.relu(x)
        x = self.fc7(x)
        x = F.relu(x)
        x = self.fc8(x)

        return x


if __name__ == "__main__":

    # Load data
    transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(), transforms.RandomAffine((-30, 30), (0.05, 0.05)), transforms.ToTensor()])

    transform1 = transforms.Compose([transforms.ToTensor()])

    data_path = os.path.join(sys.path[0], "data")

    train_set = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True, num_workers=0)

    test_set = torchvision.datasets.MNIST(root=data_path, train=False, download=True, transform=transform1)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0)

    # Net
    net = AlexNet()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adadelta(net.parameters())
    # optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.to(device=device)

    # Training session
    print("[INFO] Start training")
    epoch_array = []
    loss_array = []

    num_epochs = 20

    for epoch in range(num_epochs):
        running_loss = 0
        batch_size = 100

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            print(outputs.size())
            print(labels.size())
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_array.append((epoch + 1) * 100)
            loss_array.append(loss.item())

            print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

    plt.plot(epoch_array, loss_array)
    plt.show()
    plt.savefig('plot.png')
    print("[INFO] Finished training")

    # Save trained model
    file_path = os.path.join(sys.path[0], "MNIST.pkl")
    torch.save(net, file_path)

    trained_model = torch.load(file_path)

    # Start testing
    with torch.no_grad():
        correct = 0
        total = 0

        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            out = trained_model(images)
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("[RESULT] Accuracy:", 100 * correct / total, "%")
