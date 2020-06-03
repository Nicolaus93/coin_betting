import torch
import torch.nn as nn
import optimal_pytorch
from optimal_pytorch import *
from examples import mnist
import cProfile
import line_profiler
import atexit
import builtins
#reproducability
torch.manual_seed(1)
#get a list of all optimizers in the module and exclude the parent class Optimizer
optimizers_list = [
    x for x in dir(optimal_pytorch)
    if x.find('__') < 0 and not x == 'Optimizer'
]


def profiler_step(conf, device, train_loader, net, optimizer, loss):
    for i, data in enumerate(train_loader, 0):
        # Take one training step.
        Xtrain, ytrain = data[0].to(device), data[1].to(device)
        # Zeroing gradients for all variables.
        optimizer.zero_grad()
        # Predictions and loss.
        ypred = net(Xtrain)
        entropy_loss = loss(ypred, ytrain)
        # Calculating gradients and taking updating weights using optimizer.
        entropy_loss.backward()
        optimizer.step()
        #view outputs
        if (i % 200 == 199):
            print('iteration %d out of %d' %
                  (i + 1, len(train_loader.dataset) // conf.batch_size))


# Running experiment on device:cpu to get cpu time
device = torch.device("cpu")
for ele in optimizers_list:
    
    # Initializing line_profiler for every optimizer
    prof = line_profiler.LineProfiler()
    print('Running 1 epoch for optimizer: ' + ele)

    # Initializing our network, loss, optimizer and training/testing data.
    net = mnist.Net().to(device)
    loss = nn.CrossEntropyLoss()
    
    if (ele.lower() == 'sgdol'):
        optimizer = getattr(optimal_pytorch, ele)(net.parameters())
    else:
        optimizer = getattr(optimal_pytorch, ele)(net.parameters(), lr=0.001)
    
    conf = mnist.Config(batch_size=60, epochs=1)
    train_loader, test_loader = mnist.prepare_data(config=conf, n_workers=0)
    
    for e in range(conf.epochs):
        lp_wrapper = prof(profiler_step)
        lp_wrapper(conf, device, train_loader, net, optimizer, loss)
        # Check the performance of network on test/validation set
        print('\n')
        prof.print_stats()
