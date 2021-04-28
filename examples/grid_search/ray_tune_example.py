from collections import defaultdict
from functools import partial
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import yaml
from ray import tune
from torch.optim import SGD, Adam
from optimal_pytorch.coin_betting.torch import Cocob
from optimal_pytorch.test_functions.loss import Absolute, Sinusoidal, Ackley


def experiment(config, loss_select, opt_select):
    """Run optimizers with all configurations from config"""
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
                "hyperparams": list(opt_params.keys()),
            }
            loss_fn = loss_select[loss]
            optimizer = opt_select[opt]
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
    """Single experiment with ray."""
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
        configuration = yaml.safe_load(stream)
    loss_dict = {"Absolute": Absolute, "Sinusoidal": Sinusoidal, "Ackley": Ackley}
    opt_dict = {"SGD": SGD, "Cocob": Cocob, "Adam": Adam}

    # run experiments
    best = defaultdict(dict)
    for result in experiment(configuration, loss_dict, opt_dict):
        single_analysis, opt_name, loss_name = result
        best[loss_name][opt_name] = single_analysis.best_dataframe

    # plot results
    for loss_name in best:
        ax = None
        names = []
        for opt_name in best[loss_name]:
            ax = best[loss_name][opt_name].avg_subopt_gap.plot(ax=ax)
            names.append(opt_name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Avg subopt gap")
        ax.set_title(f"{loss_name} loss")
        ax.legend(names)
        plt.show()
