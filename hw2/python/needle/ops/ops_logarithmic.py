from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        Z -= Z.max(-1, keepdims=True)
        return Z - array_api.log(array_api.exp(Z).sum(-1, keepdims=True))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad - node.exp() * out_grad.sum(-1, keepdims=True)
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = None if axes is None else (axes,) if isinstance(axes, int) else axes

    def compute(self, Z: NDArray):
        ### BEGIN YOUR SOLUTION
        max_Z = Z.max(self.axes, keepdims=True)
        return array_api.log(array_api.exp(Z - max_Z).sum(self.axes)) + max_Z.squeeze(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        shape = list(a.shape)
        for axis in self.axes or range(len(shape)):
            shape[axis] = 1
        out_grad = out_grad.reshape(shape).broadcast_to(a.shape)
        node = node.reshape(shape).broadcast_to(a.shape)
        return out_grad * (a - node).exp()
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

