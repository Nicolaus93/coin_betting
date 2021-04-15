import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from optimal_pytorch.optim import CoinBetting
from torch.utils.tensorboard import SummaryWriter
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def loaders(train_size=4, test_size=64, num_workers=2):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=train_size, shuffle=True, num_workers=num_workers)

    # Preparing test set in the data folder.
    testset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=test_size, shuffle=False, num_workers=num_workers)

    return trainloader, testloader


def training(net, criterion, optimizer, trainloader, testloader, writer, epochs=2):
    print("Now training..")
    running_loss = 0.0
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):

            # get the inputs;
            # data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # every 1000 mini-batches...
            if i % 1000 == 999:

                # log the running loss
                writer.add_scalar(
                    'training loss', running_loss / 1000, epoch * len(trainloader) + i)

                # log a Matplotlib Figure showing the model's predictions on a
                # random mini-batch
                writer.add_figure(
                    'predictions vs. actuals', plot_classes_preds(net, inputs, labels),
                    global_step=epoch * len(trainloader) + i)
                running_loss = 0.0
    print('Finished Training')


def images_to_probs(net, images):
    """
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    """
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    classes = [i for i in range(10)]
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


if __name__ == "__main__":
    net = Net()
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = CoinBetting(net.parameters())
    trainloader, testloader = loaders()
    writer = SummaryWriter('runs/MNIST_experiment_{}'.format(type(optimizer).__name__))
    training(net, criterion, optimizer, trainloader, testloader, writer)
