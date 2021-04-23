import torch
import matplotlib.pyplot as plt
import yaml
from ray import tune
from torch.optim import SGD, Adam
from optimal_pytorch.coin_betting.torch import Cocob
from optimal_pytorch.test_functions.loss import Absolute, Sinusoidal, Ackley
from pathlib import Path
from functools import partial
from collections import defaultdict


def experiment(config):
    losses = config["loss"]
    optimizers = config["optimizer"]
    for loss in losses:
        for opt in optimizers:
            loss_config = {**config["loss"][loss]}
            opt_params = config["optimizer"][opt]
            opt_config = {i: tune.grid_search(opt_params[i]) for i in opt_params}
            exp_config = {
                **loss_config,
                **opt_config,
                "hyperparams": [i for i in opt_params],
            }
            loss_fn = eval(loss)
            optimizer = eval(opt)
            run = partial(single_run, loss=loss_fn, opt=optimizer)
            analysis = tune.run(
                run,
                name=opt,
                local_dir=Path(__file__).parent.absolute() / "tune_results" / loss,
                metric="avg_subopt_gap",
                mode="min",
                num_samples=1,
                config=exp_config,
            )
            yield analysis, opt, loss


def single_run(config, loss, opt):
    hyperparams = {i: config[i] for i in config["hyperparams"]}
    x = torch.tensor(config["x1"]).requires_grad_()
    loss_fn = loss()
    optimizer = opt([x], **hyperparams)
    cumulative_loss = 0
    opt_loss = loss_fn(loss_fn.minimum()).item()
    for i in range(config["steps"]):
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            x.clamp_(0, 10)  # these values shoud vary for each loss
        # Feed the score back back to Tune.
        cumulative_loss += loss.item()
        tune.report(iterations=i, avg_subopt_gap=cumulative_loss / (i + 1) - opt_loss)


if __name__ == "__main__":

    # read config file
    config_file = Path(__file__).parent.absolute() / "config.yaml"
    with config_file.open() as stream:
        config = yaml.safe_load(stream)

    # run experiments
    best = defaultdict(dict)
    for result in experiment(config):
        analysis, opt, loss = result
        best[loss][opt] = analysis.best_dataframe

    # plot results
    for loss in best:
        ax = None
        names = []
        for opt in best[loss]:
            ax = best[loss][opt].avg_subopt_gap.plot(ax=ax)
            names.append(opt)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Avg subopt gap")
        ax.set_title(f"{loss} loss")
        ax.legend(names)
        plt.show()
