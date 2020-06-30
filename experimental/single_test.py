import sys
import os
import torch
import numpy as np
import optimal_pytorch
from ray import tune
from ray.tune import CLIReporter
from functools import partial
from test_optimizers_mp import generate_functions


# opt_params = {
#     'Adam': {
#         'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
#         'betas': [(0.9, 0.999), (0, 0.99)]
#     },
#     'SGD': {
#         'lr': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
#         'momentum': [0.1, 0.5, 0.9, 0.999],
#         'nesterov': [False, True]
#     },
#     'SGDOL': {
#         'smoothness': [10, 20],
#         'alpha': [10, 20]
#     }
# }

func_params = ['fprime', 'fs', 'xs', 'xe', 'mu', 'sd']
func_constraints = {
    'loss_x_sinx': {
        'xs': 0,
        'xe': 10
    },
    'loss_synthetic_func': {
        'xs': 0,
        'xe': 1
    },
    'loss_gaussian': {
        'xs': 1,
        'xe': 2
    }
}


def single_run(loss_function, optimizer, initial, config, iterations=100):

    x = initial.requires_grad_()
    opt = getattr(optimal_pytorch, optimizer)([x], **config)
    for i in range(iterations):
        opt.zero_grad()
        loss = loss_function.forward(x)
        loss.backward()
        opt.step()
    minimum = loss_function.get_minima()
    return float(np.abs(minimum - x.detach().numpy())[0])


def exp(config, runs=10):
    sys.path.append(os.path.abspath(os.path.join('..', 'tests')))
    import functions

    functions_list = [x for x in dir(functions) if x.find('loss_') >= 0]
    func_parameter_combinations = generate_functions(
        functions_list,
        func_params,
        func_constraints)

    name = "loss_absolute"
    params = func_parameter_combinations[name]
    loss = getattr(functions, name)(params)
    for i in range(runs):
        x1 = torch.rand(1, dtype=torch.float) * (loss.xe - loss.xs) + loss.xs
        d = single_run(loss, "SGD", x1, config)
        tune.report(mean_dist=d)


if __name__ == "__main__":
    search_space = {
        "lr": tune.grid_search([0.001, 0.01, 0.1]),
        "momentum": tune.grid_search([0.1, 0.5, 0.9, 0.999]),
        "nesterov": tune.grid_search([False, True])
    }

    # Limit the number of rows.
    reporter = CLIReporter(max_progress_rows=10)
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    reporter.add_metric_column("mean_dist")

    # run with tune
    current_test = partial(exp, runs=10)
    analysis = tune.run(
        current_test, config=search_space, local_dir="test_experiment", progress_reporter=reporter)
    print("Best config: ", analysis.get_best_config(metric="mean_dist", mode='min'))

    # Get a dataframe for analyzing trial results.
    df = analysis.dataframe()
