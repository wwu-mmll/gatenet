import numpy as np
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def squared(x):
    return x ** 2


def func_softmax(x, dim, func):
    """
    A generalized version of the softmax function where the exponential is replaced by 'func', i.e.:

        func_softmax(x, dim, func) = func(x)/sum(func(x), dim)

    """
    f_x = func(x)
    norm = torch.sum(f_x, dim, keepdim=True)
    norm = norm.expand(f_x.shape)

    return f_x / norm


def squared_difference(x, y):
    '''
    Input: x is a bxNxd matrix
           y is a bxMxd matirx
    Output: dist is a bxNxM matrix where dist[n,i,j] is the square norm between x[n,i,:] and y[n,j,:].
    i.e. dist[b,i,j] = ||x[b,i,:]-y[b,j,:]||^2
    '''
    x_norm = (x ** 2).sum(-1).view(x.shape[0], -1, 1)
    y_norm = (y ** 2).sum(-1).view(y.shape[0], 1, -1)

    y_t = torch.transpose(y, 1, 2)

    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    return dist


class MAB(nn.Module):
    """
    Multihead attention Block (MAB). Performs multihead attention with a residual connection followed by
    a row-wise feedworward layer with a residual connection:

        MAB(X,Y) = LayerNorm(H(X,Y) + rFF(H(X,Y)))

    where

        H(X,Y) = LayerNorm(X + Multihead(X, Y, Y))

    for matrices X, Y. The three arguments for Multihead stand for Query, Value and Key matrices.
    Setting X = Y i.e. MAB(X, X) would result in the type of multi-headed self attention that was used in the original
    transformer paper 'attention is all you need'.
    Furthermore in the original transformer paper a type of positional encoding is used which is not present here.
    """

    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()

        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)

        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), dim=0)
        K_ = torch.cat(K.split(dim_split, 2), dim=0)
        V_ = torch.cat(V.split(dim_split, 2), dim=0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)

        return O


class SAB(nn.Module):
    """
    The Set Attention Block (SAB) is defined as

        SAB(X) := MAB(X,X)

    i.e. standard self attention as described in the 'attention is all you need' paper without positional encoding.
    """

    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()

        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    """
    The Induced Set Attention Block (ISAB) uses learnable 'inducing points' to reduce the complexity from O(n^2)
    to O(nm) where m is the number of inducing points. While the number of these inducing points is a fixed parameter
    the points themselves are learnable parameters.
    The ISAB is then defined as

        ISAB(X) = MAB(X,H)

    where

        H = MAB(I,X)

    i.e. ISAB(X) = MAB(X, MAB(I, X)).
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()

        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)

        return self.mab1(X, H)


class PISAB(nn.Module):
    """
    The Induced Set Attention Block (ISAB) uses learnable 'inducing points' to reduce the complexity from O(n^2)
    to O(nm) where m is the number of inducing points. While the number of these inducing points is a fixed parameter
    the points themselves are learnable parameters.
    The ISAB is then defined as

        ISAB(X) = MAB(X,H)

    where

        H = MAB(I,X)

    i.e. ISAB(X) = MAB(X, MAB(I, X)).
    """

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(PISAB, self).__init__()

        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)
        self.pma = PMA(dim_in, num_heads, num_inds)

    def forward(self, X):
        I = self.pma(X)
        H = self.mab0(I.repeat(X.size(0), 1, 1), X)

        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class LinModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        dim_output = 1
        dim_input = kwargs['_num_markers']

        self.model = nn.Sequential(
            nn.Linear(dim_input, dim_input),
            nn.ReLU()
        )
        self.head = nn.Linear(dim_input, dim_output)

    def forward(self, x):
        y = self.model(x)

        return self.head(y)[:, :, 0]


class BaseModel(nn.Module):
    """
    Base class for all models.
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class SetTransformerEncoder(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """

    def __init__(self, **kwargs):
        super(SetTransformerEncoder, self).__init__()

        dim_input = kwargs['_num_markers']
        if 'cluster' in kwargs:
            if kwargs['cluster']:
                dim_input = dim_input + dim_input ** 2 + 1
        self.sequence_length = sequence_length = kwargs['_sequence_length']
        num_inds = kwargs['num_inds']
        dim_hidden = kwargs['dim_hidden']  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads = kwargs['num_heads']
        ln = kwargs['layer_norm']

        enc_layers = [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)]
        for _ in range(1, kwargs['hidden_layers']):
            enc_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        enc_layers.append(
            ISAB(dim_hidden, dim_input, 1, num_inds, ln=ln))  # num_heads == 1 because dim_input can be a prime number
        self.enc = nn.Sequential(*enc_layers)

        # enc_layers = [ISAB_dist(dim_input, dim_hidden, num_heads, num_inds, ln=ln)]
        # for _ in range(1, kwargs['hidden_layers']):
        #     enc_layers.append(ISAB_dist(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        # enc_layers.append(ISAB_dist(dim_hidden, dim_input, 1, num_inds, ln=ln)) #num_heads == 1 because dim_input can be a prime number
        # self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        return self.enc(x)


class SetTransformer(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    dim_output: dimensionality of output             (flowdata: 1)
    dim_hidden: dimension of hidden representation
    ln:         use layer norm true/false
    """

    def __init__(self, **kwargs):
        super(SetTransformer, self).__init__()

        dim_input = kwargs['_num_markers']
        if 'cluster' in kwargs:
            if kwargs['cluster']:
                dim_input = dim_input + dim_input ** 2 + 1
        dim_hidden = kwargs['dim_hidden']  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        self.mode = kwargs['mode']
        self.residual = kwargs['residual']

        if self.mode == 'autoencoder':
            # dim_output = dim_input
            dim_output = kwargs['dim_output']
        elif self.mode == 'binary':
            dim_output = 1
        else:
            raise NotImplementedError

        assert not (
                    self.mode == 'autoencoder' and self.residual), f'mode is set to {self.mode} while residual connections are used!'

        self.enc = SetTransformerEncoder(**kwargs)
        if self.residual:
            dec_layers = [nn.Linear(2 * dim_input, dim_output)]
        else:
            dec_layers = [nn.Linear(dim_input, dim_output)]
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        enc_out = self.enc(x)

        if self.residual:
            dec_in = torch.cat([enc_out, x], dim=-1)
        else:
            dec_in = enc_out

        if self.mode == 'autoencoder':
            return self.dec(dec_in)
        elif self.mode == 'binary':
            return self.dec(dec_in)[:, :, 0]


class SetTransformerEncoderBottleneck(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    dim_output: dimensionality of output             (flowdata: 1)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """

    def __init__(self, **kwargs):
        super(SetTransformerEncoderBottleneck, self).__init__()

        dim_input = kwargs['_num_markers']
        self.sequence_length = sequence_length = kwargs['_sequence_length']
        num_inds = kwargs['num_inds']
        dim_hidden = kwargs['dim_hidden']  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
        num_heads = kwargs['num_heads']
        ln = kwargs['layer_norm']

        enc_layers = [ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)]
        for _ in range(1, kwargs['hidden_layers']):
            enc_layers.append(ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        enc_layers.append(
            ISAB(dim_hidden, dim_input, 1, num_inds, ln=ln))  # num_heads == 1 because dim_input can be a prime number
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        return self.enc(x)


if __name__ == '__main__':
    m = SetTransformer(num_inds=16, dim_hidden=32, num_heads=4, layer_norm=True, hidden_layers=3,
                       residual=False, mode="binary", _num_markers=11, _sequence_length=False)
    x__ = torch.rand(1, 1000000, 11)
    y__ = m(x__)
    #print(m)
