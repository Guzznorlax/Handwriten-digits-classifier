import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import sys
import matplotlib.pyplot as plt
from alexnet import AlexNet


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
