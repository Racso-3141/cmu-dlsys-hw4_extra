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
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).reshape((out_features,)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        without_bias = X@self.weight
        if not hasattr(self, 'bias'):
            return without_bias
        return without_bias + self.bias.reshape((1, self.out_features)).broadcast_to(without_bias.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return X.reshape((X.shape[0], int(np.prod(X.shape[1:]))))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = x
        for module in self.modules:
            output = module(output)
        return output
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_examples, num_classes = logits.shape
        one_hot_y = init.one_hot(num_classes, y, device=logits.device, dtype=logits.dtype)
        z = ops.summation(one_hot_y*logits, axes=(1,)).reshape((-1, 1))
        log_sum_exp = ops.logsumexp(logits, axes=(1,)).reshape((-1, 1))
        return ops.summation(log_sum_exp - z, ) / num_examples
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        b = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)

        if self.training:
            num_examples = x.shape[0]

            # Calculate batch mean
            mean = ops.summation(x, axes=(0,)) / num_examples

            # Calculate batch variance
            diff = x - mean.reshape((1, self.dim)).broadcast_to(x.shape)
            var = ops.summation(diff ** 2, axes=(0,)) / num_examples

            # Update running statistics
            self.running_mean = (
                self.running_mean * (1 - self.momentum)
                + mean.detach() * self.momentum
            )

            self.running_var = (
                self.running_var * (1 - self.momentum)
                + var.detach() * self.momentum
            )

        else:
            # Use running statistics during evaluation
            mean = self.running_mean
            var = self.running_var

        # Broadcast mean and variance
        mean = mean.reshape((1, self.dim)).broadcast_to(x.shape)
        var = var.reshape((1, self.dim)).broadcast_to(x.shape)

        # Normalize
        normalized = (x - mean) * ((var + self.eps) ** -0.5)

        # Apply learnable scale and bias
        return normalized * w + b


class BatchNorm2d(BatchNorm1d):
    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x_nhwc = x.transpose((1, 2)).transpose((2, 3))
        flattened = x_nhwc.reshape((n * h * w, c))
        normalized = super().forward(flattened)
        return normalized.reshape((n, h, w, c)).transpose((2, 3)).transpose((1, 2))



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(len(x.shape) - 1, )) / self.dim
        mean = mean.reshape((-1, 1)).broadcast_to(x.shape)
        diff = x - mean
        
        var = ops.summation(diff ** 2, axes=(len(x.shape) - 1,)) / self.dim
        var = var + self.eps
        
        std = var ** 0.5
        std = std.reshape((-1, 1)).broadcast_to(x.shape)

        w = self.weight.reshape((1, -1)).broadcast_to(x.shape)
        b = self.bias.reshape((1, -1)).broadcast_to(x.shape)

        return diff / std * w + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1-self.p, device=x.device, dtype=x.dtype)
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION