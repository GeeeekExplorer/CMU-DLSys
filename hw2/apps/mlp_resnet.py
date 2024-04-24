import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(nn.Sequential(
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim)
        )),
        nn.ReLU()
    )
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt is not None:
        model.train()
        opt.reset_grad()
    else:
        model.eval()
    softmax_loss = nn.SoftmaxLoss()
    sum_loss, sum_err, total = 0, 0, len(dataloader.dataset)
    for batch in dataloader:
        X, y = batch
        logit = model(X)
        loss = softmax_loss(logit, y)
        # print(p)
        loss.backward()
        # print(p.grad)
        if opt is not None:
            opt.step()
            opt.reset_grad()
        # print(loss.numpy().item())
        sum_loss += loss.numpy().item() * y.shape[0]
        sum_err += (logit.numpy().argmax(-1) != y.numpy()).sum().item()
        # if i == 20:
        #     break
    return sum_err / total, sum_loss / total
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    dataset = ndl.data.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", f"{data_dir}/train-labels-idx1-ubyte.gz")
    dataloader = ndl.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(epochs):
        train_err, train_loss = epoch(dataloader, model, opt)
    dataset = ndl.data.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz", f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    dataloader = ndl.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    test_err, test_loss = epoch(dataloader, model, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
