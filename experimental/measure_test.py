import optimal_pytorch as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from optimal_pytorch.functions import *
from ray import tune
from ray.tune import CLIReporter
from ray.tune.analysis.experiment_analysis import Analysis
from ray.tune.schedulers import hyperband
from functools import partial
from typing import Mapping, Sequence, NewType, Any
from pathlib import Path
from collections import Counter

opt_algo = NewType('opt_algo', optim.optimizer)


def suboptimality_gap(
    loss_function: GenericLoss,
    optimizer: optim.optimizer,
    initial: torch.tensor,
    config: Mapping[str, float],
    iterations: int = 100
) -> float:

    x = initial.requires_grad_()
    # initialize optimizer
    opt = optimizer([x], **config)
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


def experiment(
    optimizer: optim.optimizer,
    loss_f: GenericLoss,
    configuration: Mapping[str, Any],
    runs: int = 10,
) -> None:
    """ Run experiment with ray.tune """
    torch.manual_seed(42)  # fix seed
    for i in range(runs):
        x1 = torch.rand(1, dtype=torch.float) * (loss_f.xe - loss_f.xs) + loss_f.xs
        eps = suboptimality_gap(loss_f, optimizer, x1, configuration)
        tune.report(subopt_gap=eps)


def run_algos(
    optimizers: Sequence[opt_algo],
    loss_function: GenericLoss,
    name: str = None
) -> Mapping[str, Mapping[str, float]]:
    # define tune reporter
    reporter = CLIReporter(max_progress_rows=10)
    try:
        # Add a custom metric column, in addition to the default metrics.
        # Note that this must be a metric that is returned in your training results.
        reporter.add_metric_column("subopt_gap")
    except ValueError:
        pass
    # store results
    results = dict()
    # run algorithms
    for algo in optimizers:
        # retrieve parameters to try
        params = algo.grid_search_params()
        # define search space in tune format
        search_space = {key: tune.grid_search(params[key]) for key in params}
        # define next experiment
        current_test = partial(experiment, algo, loss_function, runs=10)
        # define dir to save results
        path = Path("optimizers") / loss.name / algo.__name__
        if (path / name).exists():
            # skip this run
            print("Experiment already run, skipping it..")
            analysis = Analysis(path)
        else:
            # run with tune
            analysis = tune.run(current_test, config=search_space, local_dir=path,
                                progress_reporter=reporter, name=name)
        results[algo.__name__] = analysis.get_best_config(
            metric="subopt_gap", mode="min")
    return results


def retrieve_results(results_path: Path) -> Mapping[str, float]:
    best_version = dict()
    for opt in results_path.iterdir():
        if opt.name.startswith('.'):
            continue
        analysis = Analysis(opt)
        p = Path(analysis.get_best_logdir(metric="subopt_gap", mode="min"))
        results = p / "progress.csv"
        df = pd.read_csv(results)
        best_version[opt.name] = df["subopt_gap"].mean()
    return best_version


def printProgressBar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 100,
    fill: str = "█",
):
    """
    Call in a loop to create terminal progress bar
    as explained here https://stackoverflow.com/questions/3160699/python-progress-bar
    @params:
        iteration   - current iteration
        total       - total iterations
        prefix      - prefix string
        suffix      - suffix string
        decimals    - positive number of decimals in percent complete
        length      - character length of bar
        fill        - bar fill character
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
    name = "find_best"
    sched = False

    verbose = False
    plot = False
    if plot:
        x = np.linspace(1, 10, num=100, endpoint=True, retstep=False, dtype=None)
        y = [loss.forward(i) for i in x]
        plt.plot(y)
        plt.show()

    opt_algos = [optim.SGD, optim.AccSGD, optim.AdaBound, optim.SGDOL]

    # run all algorithms
    for loss in losses:
        best_versions = run_algos(opt_algos, loss, name=name)
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
            try:
                if res[algo] <= res["SGD"]:
                    progress[algo] += 1
            except KeyError:
                print("Consider running tests on SGD first!")
                exit(0)

    for algo in progress:
        pref = "{:<8} - passed tests:".format(algo)
        printProgressBar(progress[algo], len(losses), prefix=pref, suffix="", length=50)
