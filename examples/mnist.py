import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from types import ModuleType
import optimal_pytorch.optim as optim
import argparse
from Data.scripts import config
# Checking if gpu exists.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#for reproducibility
torch.manual_seed(1)
# Setting REPRODUCIBLE to true will ensure same performance between cpu and gpu, but will make the program slower
REPRODUCIBLE = False
if (REPRODUCIBLE):
    if device == torch.device('cuda:0'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def prepare_data(config, n_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307, ), (0.3081, ))
    ])  # 0.1307 and 0.3081 are the mean and std of MNIST

    # Preparing training set in the data folder.
    trainset = torchvision.datasets.MNIST(
        root='../Data',
        train=True,
        download=True,
        transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=n_workers)

    # Preparing test set in the data folder.
    testset = torchvision.datasets.MNIST(root='../Data',
                                         train=False,
                                         download=True,
                                         transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=config.test_batch_size,
                                             shuffle=False,
                                             num_workers=n_workers)
    return trainloader, testloader


# Future function to be used for showing the images and classified.
def imshow(img):
    # Denormalizing image.
    img = (img * 0.3081) + 0.1307
    img = img.cpu().numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def train_step(device, model, optimizer, loss, data, i):
    # Data is just a list of images and labels.
    Xtrain, ytrain = data[0].to(device), data[1].to(device)
    # Zeroing gradients for all variables.
    optimizer.zero_grad()

    # Predictions and loss.
    ypred = model(Xtrain)
    entropy_loss = loss(ypred, ytrain) / data[0].size(0)
    # Calculating gradients and taking updating weights using optimizer.
    entropy_loss.backward()
    optimizer.step()

    return entropy_loss


# Test the model on test set.
def test_outputs(device, model, loss, test_loader, epoch, istrain=False):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            Xtest, ytest = data[0].to(device), data[1].to(device)
            yprob = model(Xtest)
            ypred = yprob.argmax(dim=1)
            test_loss += loss(yprob, ytest).item()
            correct += (ypred == ytest).sum().item()
            total += ytest.size(0)
    if (istrain):
        print('iteration', str(epoch), ' Accuracy of the network: ',
              str(correct / total), ' Training loss: ', str(test_loss / total))
    else:
        print('Epoch', str(epoch), ' Accuracy of the network: ',
              str(correct / total), ' Test loss: ', str(test_loss / total))


def main():
    parser = argparse.ArgumentParser(
        description=
        'This is a simple example which runs NN on MNIST dataset, using the optimizer provided. Current Optimizers available are :'
        + str([
            ele for ele in dir(optim)
            if (ele.find('__') < 0 and ele.find('Optimizer') < 0)
        ]))
    parser.add_argument(
        '--optimizer',
        type=str,
        help='which optimizer do you like to add(SGD is default)?')
    args = parser.parse_args()
    opt = args.optimizer
    if (not opt):
        opt = 'SGD'
    opt = opt.lower()
    try:
        if (getattr(optim, opt)):
            get_opt = getattr(optim, opt)
            if(type(get_opt)==ModuleType):
                for ele in dir(get_opt):
                    if(ele.lower()==opt):
                        opt = ele
            else:
                opt = opt
    except AttributeError:
        print('there is no opt by this name... chosing default SGD')
        opt = 'SGD'
    # Initializing our network, loss, optimizer and training/testing data.
    net = Net().to(device)
    loss = nn.CrossEntropyLoss(reduction='sum')
    if (opt == 'SGDOL'):
        optimizer = getattr(optim, opt)(net.parameters())
    else:
        optimizer = getattr(optim, opt)(net.parameters(), lr=0.001)
    conf = config.MNIST_Config()
    train_loader, test_loader = prepare_data(conf, 2)
    for e in range(conf.epochs):
        for i, data in enumerate(train_loader, 0):
            # Take one training step.
            _ = train_step(device, net, optimizer, loss, data, i)
            if (i % 200 == 199):
                test_outputs(device, net, loss, train_loader, i + 1, True)
        # Check the performance of network on test/validation set
        test_outputs(device, net, loss, test_loader, e)


if __name__ == '__main__':
    main()
