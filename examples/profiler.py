import torch
import torch.nn as nn
import optimal_pytorch.optim as optimal_pytorch
from examples import mnist
import line_profiler
from Data.scripts import config
import argparse
from tqdm import tqdm
import os
# reproducability
torch.manual_seed(1)
#get a list of all optimizers in the module and exclude the parent class Optimizer
optimizers_list = [
    ele for ele in optimal_pytorch.__all__
    if 'grid_search_params' in dir(getattr(optimal_pytorch, ele))
]


def profiler_step(conf, device, train_loader, net, optimizer, loss, verbose,
                  optim):
    """Function which runs the profiler step for 1 epoch on net with an optimizer and a loss function.
    conf : config object which contains parameters for program and neural net.
    device : device to run nn on, cuda:0 if gpu else cpu
    train_loader : iterator of training dataset.
    net : neural net instance.
    Optimizer : The optimizer used.
    Loss : loss function used.
    """
    with tqdm(total=1000, ncols=100, desc='Running ' + optim) as pbar:
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
            if (verbose):
                if (i % 200 == 199):
                    print(
                        'iteration %d out of %d' %
                        (i + 1, len(train_loader.dataset) // conf.batch_size))
            pbar.update(1)


def main(args):
    if (os.path.exists('../Results/')):
        results_path = '../Results/'
        data_location = '../Data/'
    elif (os.path.exists('./Results/')):
        results_path = './Results/'
        data_location = './Data/'

    results = {}
    # Running experiment on device:cpu to get cpu time
    device = torch.device("cpu")
    print('Currently available optimizers are : \n')
    print(optimizers_list)
    print('Running 1 Epoch for every optimizer')
    for optim in optimizers_list:
        # Initializing line_profiler for every optimizer
        prof = line_profiler.LineProfiler()

        # Initializing our network, loss, optimizer and training/testing data.
        net = mnist.Net().to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = getattr(optimal_pytorch, optim)(net.parameters())

        conf = config.MNIST_Config(batch_size=60, epochs=1)
        train_loader, test_loader = mnist.prepare_data(
            config=conf, n_workers=0, data_location=data_location)

        for e in range(conf.epochs):
            lp_wrapper = prof(profiler_step)
            lp_wrapper(conf, device, train_loader, net, optimizer, loss,
                       args.verbose, optim)
            # Check the performance of network on test/validation set
            stats = prof.get_stats()
            for (fn, lineno, name), timings in sorted(stats.timings.items()):
                timings = stats.timings[fn, lineno, name]
                # print(fn, lineno, name, stats.unit, timings)
                for ele in timings:
                    # 34 is the line number for optimizer step.
                    if (ele[0] == 34):
                        results[optim] = [ele[2], float(ele[2]) / ele[1]]
            if (args.verbose):
                prof.print_stats()
        print('\n')
    baseline = results['SGD']
    with open(results_path + 'profiler.txt', 'w+') as file:
        for key in results:
            if (key != 'SGD'):
                print(
                    '%s runs %f x times(single epoch) and %f x times SGD(per iteration) on CPU.'
                    % (key, results[key][0] / baseline[0],
                       results[key][1] / baseline[1]))
                file.write(
                    '%s runs %f x times(single epoch) and %f x times SGD(per iteration) on CPU.'
                    % (key, results[key][0] / baseline[0],
                       results[key][1] / baseline[1]))
                file.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Profile different optimizers for a single epoch on mnist.'
    )
    parser.add_argument('--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help="whether to display more information")
    args = parser.parse_args()
    main(args)
