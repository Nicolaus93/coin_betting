import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from optimal_pytorch import Recursive, Cocob


def log_loss(w, x):
    return torch.log(1 + torch.exp(w.dot(x)))


dim = 2
epochs = 2
N = 200
X, y = datasets.make_blobs(
    n_samples=N, centers=2, n_features=dim, center_box=(0, 10), random_state=42)
X = np.hstack((X, np.ones((len(X), 1))))
model = torch.zeros(dim + 1, requires_grad=True)
optimizer = Recursive([model], inner=Cocob)
criterion = nn.CrossEntropyLoss()
for _ in range(epochs):
    cum_loss = 0
    for pos, value in enumerate(zip(X, y)):
        x, label = value
        x = torch.Tensor(x)
        optimizer.zero_grad()
        if label == 0:
            x *= -1
        loss = log_loss(x, model)
        a = loss.item()
        loss.backward()
        optimizer.step()
        cum_loss += loss.item()
    print(f"Avg loss {cum_loss / N:.3f}")

plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color='orange', s=8)
plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color='blue', s=8)
w1, w2, b = model
c = -b.item() / w2.item()
m = -w1.item() / w2.item()
xmin, xmax = 0, 10
ymin, ymax = 3, 12
xd = np.array([xmin, xmax])
yd = m * xd + c
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')
plt.plot()
plt.show()
