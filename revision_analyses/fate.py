import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from abc import abstractmethod


def reset(value: Any):
    if hasattr(value, 'reset_parameters'):
        value.reset_parameters()
    else:
        for child in value.children() if hasattr(value, 'children') else []:
            reset(child)


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

    def _compute_forward(self, Q, K):
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

        return O, A

    def forward(self, Q, K):
        O, A = self._compute_forward(Q, K)

        return O


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


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()

        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class EventEmbedderType(nn.Module):
    """
    The module that embeds the Event in a general Space - using one seed vector of target dimension.

    """

    def __init__(self,
                 pos_encoding_dim: int,
                 dim_event_embedding: int,
                 num_heads_embedder: int,
                 layer_norm: bool = False):
        super().__init__()

        self.eventwise_marker_attention = MAB(1 + pos_encoding_dim, 1 + pos_encoding_dim, dim_event_embedding,
                                              num_heads=num_heads_embedder,
                                              ln=layer_norm)  # self attention # 2 is dim input since we have marker value and pos encoding, now it is marker value plus
        self.eventwise_embedder = PMA(dim_event_embedding, num_heads=num_heads_embedder, num_seeds=1, ln=layer_norm)

    def forward(self, x, pos_encodings):
        # x has dim n_events x n_marker
        # pos_encodings is matrix of shape n_events x n_marker x pos_embedding_dim

        # split in single events -> batch dim(= n_events) x n_marker x 1; transpose faster than cat
        # x_eventwise = x.unsqueeze(0).transpose(1, 2).transpose(0, 2)
        x_eventwise = x.unsqueeze(0).transpose(1, 2).transpose(0, 2)

        # add pos encoding of marker to x_eventwise -> n_events (= e.g. 600) is batchsize, n_marker = 14 and pos_enc_dim+1 = 11 -> [600, 14,11]
        x_eventwise_pos_m = torch.cat((x_eventwise, pos_encodings), dim=2)

        # Eventwise self-attention between marker -> n_events x n_marker x embedding_dim: [600, 14,2]  if embedding dim = 2
        x_embedded_ = self.eventwise_marker_attention(x_eventwise_pos_m, x_eventwise_pos_m)

        # Cross attention with seed vector -> n_events x 1=num_seeds x hidden_dim [600, 1, 2]
        x_embedded_ = self.eventwise_embedder(x_embedded_)

        # reshape
        x_embedded = x_embedded_.transpose(0, 1)  # 1 * n_events *n_marker
        return x_embedded


class FATE(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """

    def __init__(self,
                 dim_event_embedding=32,
                 num_heads_embedder=1,
                 dim_hidden=32,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
                 num_heads=4,
                 num_inds=16,
                 layer_norm=True,
                 pos_encoding_dim: int = 10  # runter drehen für memory einsparnisse,
                 ):
        super().__init__()

        # event embedder
        self.event_embedder = EventEmbedderType(pos_encoding_dim, dim_event_embedding, num_heads_embedder,
                                                layer_norm=True)

        # normal ST
        self.isab1 = ISAB(dim_event_embedding, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.isab2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=layer_norm)
        self.isab3 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=False)

    def forward(self, x, pos_encodings):
        if len(x.shape) == 3:
            x.squeeze(0)
        x_embedded = self.event_embedder(x, pos_encodings)  # 1 * n_events*n_marker
        o1 = self.isab1(x_embedded)
        o2 = self.isab2(o1)
        o3 = self.isab3(o2)  # 1* n_events * dim_output
        output = o3.squeeze(0)  # n_events * dim_hidden

        return output


class FATEMaskedAEDecoder(BaseModel):
    """
    Set transformer as described in https://arxiv.org/abs/1810.00825
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    num_inds:   number of induced points
    dim_hidden: dimension of hidden representation
    num_heads:  number of attention heads
    ln:         use layer norm true/false
    """

    def __init__(self,
                 dim_latent,
                 layer_norm,
                 dim_hidden=32,  # dim_hidden must be divisible by num_heads i.e. dim_hidden%num_heads = 0
                 num_heads=4,
                 num_inds=16,
                 pos_encoding_dim: int = 10  # runter drehen für memory einsparnisse
                 ):
        super().__init__()

        # for attention across events - either relu former without lin embedding befor or channelwise attention
        # latents attention
        self.latent_attention_1 = ISAB(dim_latent, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)
        self.latent_attention_2 = ISAB(dim_hidden, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)
        self.latent_attention_3 = ISAB(dim_hidden, dim_hidden, num_heads=num_heads, num_inds=num_inds, ln=layer_norm)

        # with the current implementation of multiple heads, more than one head does not make sense!!
        # since query is split up.. well we have alinear layer befor.. so actuall coudl make sense after all..
        self.cross_attention = MAB(pos_encoding_dim, 1, dim_hidden, num_heads=num_heads,
                                   ln=False)  # ln=layer_norm) # dim Q, dim K, dim V - q= queries (marker queries)
        self.eventwise_marker_attention = MAB(dim_hidden, dim_hidden, dim_hidden, num_heads=1, ln=False)
        self.out_ly = nn.Linear(dim_hidden, 1)

    def forward(self, pos_encodings, latents):
        n_events = pos_encodings.shape[0]

        lats1 = self.latent_attention_1(latents.unsqueeze(0))
        lats2 = self.latent_attention_2(lats1)
        lats3 = self.latent_attention_3(lats2)

        latents_eventwise = lats3.transpose(1, 2).transpose(0, 2)

        # get the marker corresponding queries to reconstruct
        # this is a memory bottle neck - since batchsize = n_events,we could split it up here
        if n_events > 50000:
            x = self.cross_attention(pos_encodings[:50000, :, :], latents_eventwise[:50000, :, :])
            n_chunks = int(n_events / 50000)  # int does floor
            for i in range(1, n_chunks):
                x = torch.cat((x, self.cross_attention(pos_encodings[50000 * i:50000 * (i + 1), :, :],
                                                       latents_eventwise[50000 * i:50000 * (i + 1), :, :])), dim=0)
            x = torch.cat((x, self.cross_attention(pos_encodings[50000 * n_chunks:, :, :],
                                                   latents_eventwise[50000 * n_chunks:, :, :])), dim=0)
        else:
            x = self.cross_attention(pos_encodings, latents_eventwise)  # n_events, n_marker, dim_hidden

        # if still too little - can split this up as well
        x_2 = self.eventwise_marker_attention(x, x)  # n_events, n_marker, dim_hidden
        out = self.out_ly(x_2).squeeze(-1)  # n_events, n_marker, 1
        return out


class SupervisedModel(BaseModel):

    def __init__(self,
                 encoder,
                 pred_head,
                 n_marker,
                 pos_encoding_dim=10,
                 encoder_out_dim=32,
                 latent_dim=8):
        super().__init__()
        self.encoder = encoder

        self.pos_encoding = torch.nn.Parameter(torch.randn(n_marker, pos_encoding_dim))
        self.fc_mu = torch.nn.Linear(encoder_out_dim, latent_dim)

        self.pred_head = pred_head
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.pred_head)

    def forward(self, x, marker_idx_encoder):
        """Returns the latent space for the input arguments, their
        corruptions and their summary representation."""

        encoded_x = self.encoder(x, self.pos_encoding[marker_idx_encoder])

        z = self.fc_mu(encoded_x)

        return self.pred_head(z), z


class MLP(BaseModel):
    """
    simple MLP - used as prediction head for weakly supervised DGI
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    dim_hidden: dimension of hidden representation
    """

    def __init__(self, dim_input=8,
                 dim_hidden=8,
                 hidden_layers=1,
                 dim_output=1,
                 skip_con: bool = False):
        super().__init__()

        mlp_layers = [nn.Linear(dim_input, dim_hidden),
                      nn.GELU()]
        for _ in range(0, hidden_layers):
            mlp_layers.extend([nn.Linear(dim_hidden, dim_hidden),
                               nn.GELU()])
        mlp_layers.append(nn.Linear(dim_hidden, dim_output))
        self.mlp = nn.Sequential(*mlp_layers)

        self.skip_con = skip_con

    def forward(self, x):
        output = self.mlp(x)
        if self.skip_con:
            output += x

            return output
        return output


class LinLayer(BaseModel):
    """
    simple MLP - used as prediction head for weakly supervised DGI
    dim_input:  dimensionality of input              (flowdata: number of markers)
    num_ouputs: output sequence length               (flowdata: sequence length)
    dim_hidden: dimension of hidden representation
    """

    def __init__(self, dim_input,
                 dim_output,
                 use_bias: bool = True):
        super().__init__()

        self.mlp = nn.Linear(dim_input, dim_output, bias=use_bias)

    def forward(self, x):
        output = self.mlp(x)

        return output


if __name__ == '__main__':
    fate = FATE()
    mlp = MLP()
    sm = SupervisedModel(encoder=fate, pred_head=mlp, n_marker=10)
    # x__ = torch.rand(1, 1000000, 11)
    # y__ = m(x__)
    #print(m)
