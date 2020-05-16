import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from optimal_pytorch import Adam, SGD, SGDOL

class Config():
    def __init__(self, batch_size=60, test_batch_size=1000, lr=1e-3, epochs=50):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.lr = lr
        self.epochs = epochs

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*5*5, 128)
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
        x = F.log_softmax(x, dim=1)
        return x

def prepareData(config):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    #preparing training set in the data folder
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=2)

    #preparing test set in the data folder
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

#future function to be used for showing the images and classified
def imshow(img):
    #denormalizing image
    img = (img*0.3081) + 0.1307
    img = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def trainStep(device, model, optimizer, loss, data):
    #data is just a list of images and labels
    Xtrain, ytrain = data[0].to(device), data[1].to(device)
    #zeroing gradients for all variables
    optimizer.zero_grad()
    
    #predictions and loss
    ypred = model(Xtrain)
    entropy_loss = loss(ypred, ytrain)
    #calculating gradients and taking updating weights using optimizer
    entropy_loss.backward()
    optimizer.step()

    return entropy_loss

def test_outputs(device, model, loss, test_loader, epoch):
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            Xtest, ytest = data[0].to(device), data[1].to(device)
            yprob = model(Xtest)
            ypred = yprob.argmax(dim=1)
            test_loss+= loss(yprob, ytest).item()
            correct+= (ypred==ytest).sum().item()
            total+= ytest.size(0)
    print('Epoch', str(epoch), ' Accuracy of the network: ', str(correct/total), ' Test loss: ', str(test_loss))

def main():
    #checking if gpu exists
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #initializing our network, loss, optimizer and training/testing data
    net = Net().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.001)
    conf = Config()
    trainloader, testloader = prepareData(conf)
    for e in range(conf.epochs):
        running_loss = 0
        for i, data in enumerate(trainloader, 0):
            #take one training step
            entropy_loss = trainStep(device, net, optimizer, loss, data)
            running_loss+=entropy_loss.item()
            if(i%200==199):
                print('Epoch: ' , str(e), ' Iteration: ', str(i), ' Training loss: ', str(running_loss/200))
                running_loss = 0
        #check the performance of network on test/validation set
        test_outputs(device, net, loss, testloader, e)

if __name__=='__main__':
    main()





    