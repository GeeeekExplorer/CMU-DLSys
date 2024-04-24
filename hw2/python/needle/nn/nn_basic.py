"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        self.bias = None
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        Y = X @ self.weight
        if self.bias:
            Y = Y + self.bias.broadcast_to(Y.shape)
        return Y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape([X.shape[0], -1])
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x.relu()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        y = init.one_hot(logits.shape[-1], y, device=y.device, dtype=logits.dtype)
        return (ops.logsumexp(logits, -1).sum() - (logits * y).sum()) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = x.shape
        if self.training:
            mean = x.sum(0) / shape[0]
            x = x - mean.broadcast_to(shape)
            var = (x ** 2).sum(0) / shape[0]
            x = x / (var + self.eps).broadcast_to(shape) ** 0.5
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            x = x - self.running_mean.broadcast_to(shape)
            x = x / (self.running_var + self.eps).broadcast_to(shape) ** 0.5
        return x * self.weight.broadcast_to(shape) + self.bias.broadcast_to(shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = x.shape
        if shape[-1] > 1:
            mean = x.sum(-1, keepdims=True) / shape[-1]
            x = x - mean.broadcast_to(shape)
            var = (x ** 2).sum(-1, keepdims=True) / shape[-1]
            x = x / (var + self.eps).broadcast_to(shape) ** 0.5
        else:
            x = x / x
        return x * self.weight.broadcast_to(shape) + self.bias.broadcast_to(shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training and self.p > 0:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device)
            x = x * mask / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
