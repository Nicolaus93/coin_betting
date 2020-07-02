import optimal_pytorch as optim
import numpy as np
# import matplotlib.pyplot as plt
import torch
from optimal_pytorch.functions.loss_functions_1d import absolute_loss, quadratic_loss
from ray import tune
from ray.tune import CLIReporter
from functools import partial
from typing import Mapping, Sequence
from optimal_pytorch.functions.loss_functions_1d import generic_loss


def suboptimality_gap(
    loss_function: generic_loss,
    optimizer: optim.optimizer,
    initial: torch.tensor,
    config: Mapping[str, float],
    iterations: int = 100
) -> float:

    x = initial.requires_grad_()
    # initialize optimizer
    opt = getattr(optim, optimizer)([x], **config)
    for i in range(iterations):
        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update.
        # Checkout docs of torch.autograd.backward for more details.
        opt.zero_grad()
        # Compute loss
        loss = loss_function.forward(x)
        # Use autograd to compute the backward pass.
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        opt.step()
    minimum = loss_function.get_minima()
    fx_star = loss_function.forward(minimum)
    fx_T = loss_function.forward(x)
    return float(abs(fx_star - fx_T))
    # return float(np.abs(minimum - x.detach().numpy())[0])


def experiment(
    optimizer: optim.optimizer,
    search_space: Mapping[str, float],
    runs: int = 10
) -> None:
    """ Run experiment with ray.tune """
    for i in range(runs):
        x1 = torch.rand(1, dtype=torch.float) * (loss.xe - loss.xs) + loss.xs
        eps = suboptimality_gap(loss, optimizer, x1, search_space)
        tune.report(subopt_gap=eps)


def run_all(optimizers: Mapping[str, Mapping[str, Sequence[float]]]) -> None:

    # define tune reporter
    reporter = CLIReporter(max_progress_rows=10)
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("subopt_gap")
    # run algorithms
    for opt_class in optimizers:
        # retrieve parameters to try
        params = optimizers[opt_class]
        # define search space in tune format
        search_space = {key: tune.grid_search(params[key]) for key in params}
        # define next experiment
        current_test = partial(experiment, opt_class, runs=10)
        # define dir to save results
        path = "optimizers/" + opt_class
        # run with tune
        analysis = tune.run(
            current_test, config=search_space, local_dir=path, progress_reporter=reporter)
        # log best configuration
        print("Best: ", analysis.get_best_config(metric="subopt_gap", mode='min', scope='avg'))


if __name__ == "__main__":

    settings = {'xs': 1, 'xe': 10, 'slope': 2, 'offset': 0}
    loss = absolute_loss(settings)
    loss = quadratic_loss(settings)
    print(loss)
    x = np.linspace(1, 10, num=100, endpoint=True, retstep=False, dtype=None)
    y = [loss.forward(i) for i in x]
    # plt.plot(y)
    # plt.show()

    opt_params = {
        "AccSGD": {
            "lr": [1e-3, 1e-2, 1e-1, 1],
            "kappa": [10, 100, 1000],
            "xi": [1, 10],
            "small_const": [0.1, 0.5, 0.7]
        },
        'Adam': {
            'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'betas': [(0.9, 0.999), (0, 0.99)]
        },
        'SGD': {
            'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
            'momentum': [0.1, 0.5, 0.9, 0.999],
            'nesterov': [False, True]
        },
        'SGDOL': {
            'smoothness': [10, 20],
            'alpha': [10, 20]
        }
    }

    # run all algorithms
    run_all(opt_params)
