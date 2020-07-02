import optimal_pytorch as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from optimal_pytorch.functions import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.analysis.experiment_analysis import Analysis
from functools import partial
from typing import Mapping, Sequence
from pathlib import Path
from collections import Counter


def suboptimality_gap(
    loss_function: GenericLoss,
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
        print(loss)
        # Use autograd to compute the backward pass.
        loss.backward()
        # Calling the step function on an Optimizer makes an update to its parameters
        opt.step()
    minimum = loss_function.get_minima()
    fx_star = loss_function.forward(minimum)
    fx_T = loss_function.forward(x)
    return float(abs(fx_star - fx_T))


def experiment(
    optimizer: optim.optimizer,
    loss_f: GenericLoss,
    configuration: Mapping[str, float],
    runs: int = 10,
) -> None:
    """ Run experiment with ray.tune """
    torch.manual_seed(42)  # fix seed
    for i in range(runs):
        x1 = torch.rand(1, dtype=torch.float) * (loss_f.xe - loss_f.xs) + loss_f.xs
        eps = suboptimality_gap(loss_f, optimizer, x1, configuration)
        tune.report(subopt_gap=eps)


def run_all(optimizers: Mapping[str, Mapping[str, Sequence[float]]],
            loss_function: GenericLoss) -> Mapping[str, Mapping[str, float]]:

    # define tune reporter
    reporter = CLIReporter(max_progress_rows=10)
    # Add a custom metric column, in addition to the default metrics.
    # Note that this must be a metric that is returned in your training results.
    try:
        reporter.add_metric_column("subopt_gap")
    except ValueError:
        pass
    # store results
    results = dict()
    # run algorithms
    for opt_class in optimizers:
        # retrieve parameters to try
        params = optimizers[opt_class]
        # define search space in tune format
        search_space = {key: tune.grid_search(params[key]) for key in params}
        # define next experiment
        current_test = partial(experiment, opt_class, loss_function, runs=10)
        # define dir to save results
        path = "optimizers/" + loss.name + "/" + opt_class
        # run with tune
        analysis = tune.run(current_test, config=search_space, local_dir=path,
                            progress_reporter=reporter, name="find_best")
        # log best configuration
        results[opt_class] = analysis.get_best_config(metric="subopt_gap", mode='min', scope='avg')
    return results


def retrieve_results(results_path: Path) -> Mapping[str, float]:
    best_version = dict()
    for opt in results_path.iterdir():
        analysis = Analysis(opt)
        p = Path(analysis.get_best_logdir(metric="subopt_gap", mode="min"))
        # print(p)
        results = p / "progress.csv"
        df = pd.read_csv(results)
        best_version[opt.name] = df["subopt_gap"].mean()
    return best_version


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=100,
    fill="█",
):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}")


if __name__ == "__main__":

    settings = {'xs': 1, 'xe': 10, 'slope': 2, 'offset': 0}
    losses = [
        QuadraticLoss(settings),
        AbsoluteLoss(settings),
        GaussianLoss({"xs": 1, "xe": 2}),
        SinusoidalLoss({"xs": 0, "xe": 10}),
        SyntheticLoss({"xs": 0, "xe": 1})]

    verbose = False
    plot = False
    if plot:
        x = np.linspace(1, 10, num=100, endpoint=True, retstep=False, dtype=None)
        y = [loss.forward(i) for i in x]
        plt.plot(y)
        plt.show()

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
    for loss in losses:
        best_versions = run_all(opt_params, loss)
        if verbose:
            print("Best version of each algorithm:")
            for each_best in best_versions:
                print("\t" + each_best, best_versions[each_best])

    # retrieve results
    progress = Counter()
    for loss in losses:
        if verbose:
            print("\nSuboptimality gap for algorithms on {}".format(loss))
        # retrieve results
        current_path = Path('.')
        opt_path = current_path / "optimizers" / loss.name
        res = retrieve_results(opt_path)
        for algo in res:
            if verbose:
                print("{}: {:.6f}".format(algo, res[algo]))
            if res[algo] <= res["SGD"]:
                progress[algo] += 1

    for algo in progress:
        pref = "{:<8} - passed tests:".format(algo)
        printProgressBar(progress[algo], len(losses), prefix=pref, suffix="", length=50)
