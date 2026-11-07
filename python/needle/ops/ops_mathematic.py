"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return a ** self.scalar

    def gradient(self, out_grad, node):
        return out_grad * self.scalar * power_scalar(node.inputs[0], self.scalar - 1)


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs ** 2)


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        perm = list(range(a.ndim))
        if self.axes is None:
            perm[-1], perm[-2] = perm[-2], perm[-1]
        else:
            i, j = self.axes
            perm[i], perm[j] = perm[j], perm[i]
        return a.permute(perm)

    def gradient(self, out_grad, node):
        if self.axes is None:
            axes = (len(out_grad.shape) - 2, len(out_grad.shape) - 1)
        else:
            axes = tuple(self.axes)
        return (transpose(out_grad, axes),)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.compact().reshape(self.shape)

    def gradient(self, out_grad, node):
        return (reshape(out_grad, node.inputs[0].shape),)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = tuple(
            i for i in range(len(self.shape) - len(input_shape))
        )
        axes += tuple(
            i + len(self.shape) - len(input_shape)
            for i, size in enumerate(input_shape)
            if size == 1
        )
        grad = out_grad
        for axis in sorted(axes, reverse=True):
            grad = summation(grad, axes=axis)
        return (reshape(grad, input_shape) if grad.shape != input_shape else grad,)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return a.sum(axis=self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        if self.axes is None:
            axes = tuple(range(len(input_shape)))
        elif isinstance(self.axes, int):
            axes = (self.axes,)
        else:
            axes = tuple(self.axes)
        shape = list(input_shape)
        for axis in axes:
            shape[axis] = 1
        return (broadcast_to(reshape(out_grad, tuple(shape)), input_shape), )


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a @ b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lhs_grad = out_grad @ transpose(rhs)
        rhs_grad = transpose(lhs) @ out_grad
        if lhs_grad.shape != lhs.shape:
            lhs_grad = summation(lhs_grad, axes=tuple(range(lhs_grad.ndim - lhs.ndim)))
        if rhs_grad.shape != rhs.shape:
            rhs_grad = summation(rhs_grad, axes=tuple(range(rhs_grad.ndim - rhs.ndim)))
        return lhs_grad, rhs_grad


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return a.log()

    def gradient(self, out_grad, node):
        return out_grad / node.inputs[0]


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return a.exp()

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return a.maximum(0)

    def gradient(self, out_grad, node):
        mask = node.inputs[0].realize_cached_data() > 0
        return out_grad * Tensor(mask, device=out_grad.device, requires_grad=False)


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad - out_grad * (tanh(node.inputs[0]) ** 2), )
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output_shape = list(args[0].shape)
        output_shape.insert(self.axis, len(args))
        output = array_api.empty(
            tuple(output_shape),
            device=args[0].device,
            dtype=args[0].dtype
        )
        for i, arg in enumerate(args):
            idx = [slice(None)] * len(output_shape)
            idx[self.axis] = i

            output[tuple(idx)] = arg
        return output
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (split(out_grad, axis=self.axis), )
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        item_shape = list(A.shape)
        item_shape.pop(self.axis)
        item_shape = tuple(item_shape)
        return tuple(
            A[tuple(
                i if j == self.axis else slice(None)
                for j in range(A.ndim)
            )].compact().reshape(item_shape)
            for i in range(A.shape[self.axis])
        )
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (stack(out_grad, axis=self.axis), )
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        out_shape = tuple(
            (1 + self.dilation) * s if i in self.axes else s
            for i, s in enumerate(a.shape)
        )
        out = array_api.full(out_shape, 0, device=a.device, dtype=a.dtype)
        slices = tuple(
            slice(None, None, 1 + self.dilation) if i in self.axes else slice(None)
            for i in range(len(out_shape))
        )
        out[slices] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        slices = tuple(
            slice(None, None, 1 + self.dilation) if i in self.axes else slice(None)
            for i in range(len(a.shape))
        )
        return a[slices]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class ConvInputGradient(TensorOp):
    def __init__(self, stride: int, padding: int):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B, out_grad):
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        _, H_out, W_out, _ = out_grad.shape
        H_dilated = (H_out - 1) * self.stride + 1
        W_dilated = (W_out - 1) * self.stride + 1
        dilated = array_api.full(
            (N, H_dilated, W_dilated, C_out),
            0,
            device=out_grad.device,
        )
        dilated[:, ::self.stride, ::self.stride, :] = out_grad

        padded = array_api.full(
            (N, H + 2 * self.padding, W + 2 * self.padding, C_in),
            0,
            device=A.device,
        )
        for u in range(K):
            for v in range(K):
                contribution = (
                    dilated.reshape((N * H_dilated * W_dilated, C_out))
                    @ B[u, v, :, :].compact().reshape((C_in, C_out)).permute((1, 0))
                ).reshape((N, H_dilated, W_dilated, C_in))
                padded[:, u : u + H_dilated, v : v + W_dilated, :] = (
                    padded[:, u : u + H_dilated, v : v + W_dilated, :]
                    + contribution
                )
        return padded[:, self.padding : self.padding + H, self.padding : self.padding + W, :]


class ConvWeightGradient(TensorOp):
    def __init__(self, stride: int, padding: int):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B, out_grad):
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        _, H_out, W_out, _ = out_grad.shape
        padded = A.pad(((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        grad = array_api.full(B.shape, 0, device=A.device)
        grad_out = out_grad.compact().reshape((N * H_out * W_out, C_out))
        for u in range(K):
            for v in range(K):
                patches = padded[
                    :,
                    u : u + H_out * self.stride : self.stride,
                    v : v + W_out * self.stride : self.stride,
                    :,
                ].compact().reshape((N * H_out * W_out, C_in))
                grad[u, v, :, :] = patches.permute((1, 0)) @ grad_out
        return grad


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        stride = self.stride
        padding = self.padding

        A = A.pad(((0, 0), (padding, padding), (padding, padding), (0, 0)))
        _, H_padded, W_padded, _ = A.shape
        H_out = (H_padded - K) // stride + 1
        W_out = (W_padded - K) // stride + 1
        Ns, Hs, Ws, Cs = A.strides

        windows = A.as_strided(
            (N, H_out, W_out, K, K, C_in),
            (Ns, Hs * stride, Ws * stride, Hs, Ws, Cs),
        ).compact().reshape((N * H_out * W_out, K * K * C_in))
        out = windows @ B.compact().reshape((K * K * C_in, C_out))
        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs
        return (
            ConvInputGradient(self.stride, self.padding)(A, B, out_grad),
            ConvWeightGradient(self.stride, self.padding)(A, B, out_grad),
        )
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
