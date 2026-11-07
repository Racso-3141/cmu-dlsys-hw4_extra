"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return (1 + ops.exp(-x)) ** -1
        ### END YOUR SOLUTION

class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.act = ops.tanh if nonlinearity == "tanh" else ops.relu
        self.device = device
        self.dtype = dtype
        
        self.W_ih = Parameter(init.rand(
            input_size, 
            hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            hidden_size, 
            hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        self.bias_ih = Parameter(init.rand(
            hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        )) if bias else None
        self.bias_hh = Parameter(init.rand(
            hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        )) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h is None:
            h = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        Z = X@self.W_ih + h@self.W_hh
        if self.bias:
            Z += self.bias_ih.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
            Z += self.bias_hh.reshape((1, self.hidden_size)).broadcast_to((bs, self.hidden_size))
        return self.act(Z)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.rnn_cells = [
            RNNCell(input_size if i == 0 else hidden_size, hidden_size, bias, nonlinearity, device, dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h0 is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        inputs = list(ops.split(X, 0))
        h0s = list(ops.split(h0, 0))
        outputs = []
        h_ns = []
        for layer in range(self.num_layers):
            h = h0s[layer]
            new_inputs = []
            for t in range(seq_len):
                input = inputs[t]
                h = self.rnn_cells[layer](input, h)
                new_inputs.append(h)
                if layer == self.num_layers - 1:
                    outputs.append(h)
            h_ns.append(h)
            inputs = new_inputs
        output = ops.stack(outputs, 0)
        h_n = ops.stack(h_ns, 0)
        return output, h_n
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.device = device
        self.dtype = dtype

        self.W_ih = Parameter(init.rand(
            input_size, 
            4*hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        self.W_hh = Parameter(init.rand(
            hidden_size, 
            4*hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        ))
        self.bias_ih = Parameter(init.rand(
            4*hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        )) if bias else None
        self.bias_hh = Parameter(init.rand(
            4*hidden_size, 
            low=-np.sqrt(1/hidden_size), 
            high=np.sqrt(1/hidden_size),
            device=device,
            dtype=dtype,
            requires_grad=True
        )) if bias else None
        self.sigmoid = Sigmoid()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        bs, _ = X.shape
        if h is None:
            h0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h

        Z = X@self.W_ih + h0@self.W_hh
        if self.bias:
            Z += self.bias_ih.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size))
            Z += self.bias_hh.reshape((1, 4*self.hidden_size)).broadcast_to((bs, 4*self.hidden_size))
        i, f, g, o = ops.split(Z.reshape((bs, 4, self.hidden_size)), 1)
        i = self.sigmoid(i)
        f = self.sigmoid(f)
        g = ops.tanh(g)
        o = self.sigmoid(o)
        c_out = f * c0 + i * g
        h_out = o * ops.tanh(c_out)
        return h_out, c_out
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.lstm_cells = [
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size, bias, device, dtype)
            for i in range(num_layers)
        ]
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, _ = X.shape
        if h is None:
            h0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
            c0 = init.zeros(self.num_layers, bs, self.hidden_size, device=self.device, dtype=self.dtype)
        else:
            h0, c0 = h
        inputs = list(ops.split(X, 0))
        h0s = list(ops.split(h0, 0))
        c0s = list(ops.split(c0, 0))
        outputs = []
        h_ns = []
        c_ns = []        
        for layer in range(self.num_layers):
            h_n = h0s[layer]
            c_n = c0s[layer]
            new_inputs = []
            for t in range(seq_len):
                input = inputs[t]
                h_n, c_n = self.lstm_cells[layer](input, (h_n, c_n))
                new_inputs.append(h_n)
                if layer == self.num_layers - 1:
                    outputs.append(h_n)
            h_ns.append(h_n)
            c_ns.append(c_n)
            inputs = new_inputs
        output = ops.stack(outputs, 0)
        h_n = ops.stack(h_ns, 0)
        c_n = ops.stack(c_ns, 0)
        return output, (h_n, c_n)
        ### END YOUR SOLUTION

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(init.randn(
            num_embeddings, embedding_dim,
            mean=0.0, std=1.0,
            device=device, dtype=dtype, requires_grad=True
        ))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs = x.shape
        one_hot = init.one_hot(self.num_embeddings, x, device=x.device, dtype=x.dtype)
        out = one_hot.reshape((seq_len * bs, self.num_embeddings)) @ self.weight
        return out.reshape((seq_len, bs, self.embedding_dim))
        ### END YOUR SOLUTION