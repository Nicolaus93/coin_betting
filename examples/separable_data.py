import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from functools import partial
from optimal_pytorch.coin_betting.torch import (
    Recursive,
    ONSBet,
    Cocob,
    Scinol2,
    Regralizer,
)


def log_loss(w, x):
    return torch.log(1 + torch.exp(w.dot(x)))


if __name__ == "__main__":
    # https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

    dim = 2
    epochs = 5
    N = 200
    X, y = datasets.make_blobs(
        n_samples=N, centers=2, n_features=dim, center_box=(0, 10), random_state=42
    )
    X = np.hstack((X, np.ones((len(X), 1))))
    criterion = nn.CrossEntropyLoss()
    algos = [
        ONSBet,
        Cocob,
        Regralizer,
        Scinol2,
        Recursive,
        partial(Recursive, inner=Cocob),
    ]
    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    for i, opt in enumerate(algos):
        model = torch.zeros(dim + 1, requires_grad=True)
        optimizer = opt([model])
        print(optimizer.__class__.__name__)
        for _ in range(epochs):
            cum_loss = 0
            for pos, value in enumerate(zip(X, y)):
                x, label = value
                x = torch.Tensor(x)
                if optimizer.__class__.__name__ == "Scinol2":
                    optimizer.update(x)
                optimizer.zero_grad()
                if label == 0:
                    x *= -1
                loss = log_loss(x, model)
                a = loss.item()
                loss.backward()
                optimizer.step()
                cum_loss += loss.item()
            print(f"\tAvg loss: {cum_loss / N:.3f}")

        ax = axes[i]
        ax.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color="orange", s=8)
        ax.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color="blue", s=8)
        w1, w2, b = model
        c = -b.item() / w2.item()
        m = -w1.item() / w2.item()
        xmin, xmax = 1, 10
        ymin, ymax = 3, 12.5
        xd = np.array([xmin, xmax])
        yd = m * xd + c
        ax.fill_between(xd, yd, ymin, color="tab:blue", alpha=0.2)
        ax.fill_between(xd, yd, ymax, color="tab:orange", alpha=0.2)
        ax.plot(xd, yd, "k", lw=1, ls="--")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.title.set_text(optimizer.__class__.__name__)

    plt.tight_layout()
    plt.show()
