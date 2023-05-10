import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size: int, heads: int):
        """
        Parameters
        :embed_size: Embedding size
        :heads: Number of attention heads
        :values: W_v * x_i
        :keys: W_k * x_i
        :queries: W_q * x_i
        """
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dimension = embed_size // heads

        assert (
            self.head_dimension * heads == embed_size
        ), f"{embed_size} needs to be divisible by {heads}"

        # nn.Linear is a linear transformation with bias of form y = xA^T + b
        self.W_values = nn.Linear(
            in_features=self.head_dimension,  # x
            out_features=self.head_dimension,  # y
            bias=False,  # b = 0
        )

        self.W_keys = nn.Linear(
            in_features=self.head_dimension,
            out_features=self.head_dimension,
            bias=False,
        )

        self.W_queries = nn.Linear(
            in_features=self.head_dimension,
            out_features=self.head_dimension,
            bias=False,
        )

        # Fully connected layer to project the outputs of the attention heads
        self.fc_out = nn.Linear(
            in_features=heads * self.head_dimension, out_features=embed_size
        )

    def forward(self, values, keys, query, mask):
        """
        Parameters
        :values: Values to be passed to the attention layer
        :keys: Keys to be passed to the attention layer
        :query: Query to be passed to the attention layer
        :mask: Mask to be passed to the attention layer
        """
        N = query.shape[0]  # Number of training examples
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dimension)
        keys = keys.reshape(N, key_len, self.heads, self.head_dimension)
        queries = query.reshape(N, query_len, self.heads, self.head_dimension)

        dot = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries shape: (N, query_len, heads, head_dimension)
        # keys shape: (N, key_len, heads, head_dimension)
        # dot shape: (N, heads, query_len, key_len)
