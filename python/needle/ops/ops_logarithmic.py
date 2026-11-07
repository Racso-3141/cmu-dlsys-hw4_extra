from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        M = Z.max(axis=self.axes, keepdims=True)
        if self.axes is None:
            M = M.reshape((1,) * len(Z.shape))
        M = M.broadcast_to(Z.shape)
        return (Z - M).exp().sum(axis=self.axes).log() + Z.max(axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        z_shape_len = len(Z.shape)
        axes = self.axes
        if axes is None:
            axes = tuple(range(z_shape_len))
        shape = tuple([1 if i in axes else Z.shape[i] for i in range(z_shape_len)])    
        
        # softmax = exp(Z - logsumexp(Z)); reuse the forward output (node)
        lse = node.reshape(shape).broadcast_to(Z.shape)
        softmax = exp(Z - lse)
        return (out_grad.reshape(shape).broadcast_to(Z.shape) * softmax, )
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

