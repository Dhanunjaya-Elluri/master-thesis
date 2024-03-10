#!/usr/bin/env python
# coding: utf-8

"""Embedding layers for Pyraformer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import math

import torch
import torch.nn as nn
from torch import Tensor


class PositionalEmbedding(nn.Module):
    """
    PositionalEmbedding implements a positional encoding layer as described in the Transformer model.

    The positional encodings are computed once and stored in a buffer. This encoding adds information
    about the relative or absolute position of the tokens in the sequence.

    Args:
        d_model (int): The dimensionality of the model's input and output.
        max_len (int): The maximum length of the input sequences. Default: 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the PositionalEmbedding layer.

        Args:
            x (Tensor): The input tensor for which to add positional encodings.

        Returns:
            Tensor: The input tensor with added positional encodings.
        """
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    """
    TokenEmbedding creates an embedding layer for tokens using a 1D convolutional neural network.

    This embedding layer can be used in models that handle sequential data, such as in natural
    language processing tasks. It maps input tokens to a higher dimensional space defined by `d_model`.

    Args:
        c_in (int): The number of input channels (e.g., features of the input tokens).
        d_model (int): The number of output channels (e.g., the dimensionality of the embedding).
    """

    def __init__(self, c_in: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        # Adjust padding based on PyTorch version
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )

        # Initialize the weights of the convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TokenEmbedding layer.

        Args:
            x (Tensor): The input tensor representing tokens.

        Returns:
            Tensor: The embedded tokens after applying the 1D convolutional layer.
        """
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    FixedEmbedding creates a fixed, non-trainable embedding layer.

    This module is useful in scenarios where the embeddings should not be modified during training,
    such as when using predetermined positional encodings. The embeddings are generated using a
    sinusoidal function, similar to the positional encoding in the Transformer model.

    Args:
        c_in (int): The size of the input dimension (e.g., the vocabulary size).
        d_model (int): The size of each embedding vector (e.g., the embedding dimension).
    """

    def __init__(self, c_in: int, d_model: int) -> None:
        super(FixedEmbedding, self).__init__()

        # Create fixed embeddings using sinusoidal functions
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        # Initialize the embedding layer with the fixed weights
        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the FixedEmbedding layer.

        Args:
            x (Tensor): The input tensor containing indices for the embeddings.

        Returns:
            Tensor: The corresponding fixed embeddings for the input indices.
        """
        return self.emb(x).detach()


class TimeFeatureEmbedding(nn.Module):
    """
    TimeFeatureEmbedding is a neural network module for embedding time-related features.

    This module takes time-related features (such as hour, day, month, etc.) as input and
    embeds them into a higher-dimensional space using a linear layer. This is useful for
    models that require a dense representation of time features.

    Args:
        d_model (int): The size of the embedding dimension.
    """

    def __init__(self, d_model: int) -> None:
        super(TimeFeatureEmbedding, self).__init__()

        # Define the input dimension for time-related features
        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the TimeFeatureEmbedding layer.

        Args:
            x (Tensor): The input tensor containing time-related features.

        Returns:
            Tensor: The embedded time features.
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    DataEmbedding is a neural network module for combining value, positional, and temporal embeddings.

    This module is designed for sequential data processing, particularly when time-related features are
    important. It combines three different embeddings:
    1. Value Embedding: Transforms the raw values to a higher dimensional space.
    2. Positional Embedding: Adds information about the position of each element in the sequence.
    3. Temporal Embedding: Embeds time-related features (like time of day, day of week).

    The output is a combination of these three embeddings, followed by dropout for regularization.

    Args:
        c_in (int): Number of input channels (features of the input data).
        d_model (int): Dimensionality of the embedding.
        dropout (float): Dropout rate for regularization. Default: 0.1.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super(DataEmbedding, self).__init__()

        # Initialize the various embeddings
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """
        Forward pass of the DataEmbedding layer.

        Args:
            x (Tensor): The input tensor representing raw values.
            x_mark (Tensor): The input tensor representing time-related features.

        Returns:
            Tensor: The combined embedding output after applying dropout.
        """
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark)
        )
        return self.dropout(x)


class CustomEmbedding(nn.Module):
    """
    CustomEmbedding is a neural network module that combines multiple embedding types.

    This module is suitable for sequential data processing and is designed to handle not only the
    raw values but also additional time-related features and sequence IDs. It combines four different
    embeddings:
    1. Value Embedding: Transforms the raw values to a higher dimensional space.
    2. Positional Embedding: Adds information about the position of each element in the sequence.
    3. Custom Temporal Embedding: Embeds custom time-related features using a linear layer.
    4. Sequence ID Embedding: Embeds sequence identifiers.

    The output is a sum of these embeddings, followed by a dropout layer for regularization.

    Args:
        c_in (int): Number of input channels (features of the input data).
        d_model (int): Dimensionality of the embedding.
        temporal_size (int): Size of the temporal feature input.
        seq_num (int): Number of unique sequence IDs.
        dropout (float): Dropout rate for regularization. Default: 0.1.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        temporal_size: int,
        seq_num: int,
        dropout: float = 0.1,
    ) -> None:
        super(CustomEmbedding, self).__init__()

        # Initialize the various embeddings
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """
        Forward pass of the CustomEmbedding layer.

        Args:
            x (Tensor): The input tensor representing raw values.
            x_mark (Tensor): The input tensor containing temporal and sequence ID features.
                             The last dimension is expected to contain sequence IDs.

        Returns:
            Tensor: The combined embedding output after applying dropout.
        """
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark[:, :, :-1])
            + self.seqid_embedding(x_mark[:, :, -1].long())
        )
        return self.dropout(x)


class SingleStepEmbedding(nn.Module):
    """
    SingleStepEmbedding is a neural network module for embedding data specifically for transformer models.

    It combines three types of embeddings:
    1. Covariate Embedding: Embeds additional covariate data.
    2. Data Embedding: Uses 1D convolution to embed the main data.
    3. Positional Embedding: Custom position encoding for transformer models.

    This module is suitable for tasks where sequential data needs to be embedded with additional
    context from covariates and sequence position.

    Args:
        cov_size (int): Size of the covariate data.
        num_seq (int): Number of unique sequence IDs.
        d_model (int): Dimensionality of the embedding.
        input_size (int): Size of the input data.
        device (torch.device): The device on which the module will operate.
    """

    def __init__(
        self,
        cov_size: int,
        num_seq: int,
        d_model: int,
        input_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size + 1, d_model)

        # Initialize the data embedding layer
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.data_emb = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )

        # Prepare position and position_vec for custom positional encoding
        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device,
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position: Tensor, vector: Tensor) -> Tensor:
        """
        Compute the transformer positional encoding.

        Args:
            position (Tensor): Position indices for the encoding.
            vector (Tensor): Vector used in calculating the positional encoding.

        Returns:
            Tensor: The computed positional encoding.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the SingleStepEmbedding layer.

        Args:
            x (Tensor): The input tensor containing main data and covariates.

        Returns:
            Tensor: The combined embedding output.
        """
        covs = x[:, :, 1 : (1 + self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(
            x[:, :, 0].unsqueeze(2).permute(0, 2, 1)
        ).transpose(1, 2)
        embedding = cov_embedding + data_embedding

        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(
            position, self.position_vec.to(x.device)
        )

        embedding += position_emb

        return embedding


class DataEmbedding_wo_pos(nn.Module):
    """
    DataEmbedding is a neural network module for combining value, and temporal embeddings.

    This module is designed for sequential data processing, particularly when time-related features are
    important. It combines three different embeddings:
    1. Value Embedding: Transforms the raw values to a higher dimensional space.
    2. Temporal Embedding: Embeds time-related features (like time of day, day of week).

    The output is a combination of these three embeddings, followed by dropout for regularization.

    Args:
        c_in (int): Number of input channels (features of the input data).
        d_model (int): Dimensionality of the embedding.
        dropout (float): Dropout rate for regularization. Default: 0.1.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super(DataEmbedding_wo_pos, self).__init__()

        # Initialize the various embeddings
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """
        Forward pass of the DataEmbedding layer.

        Args:
            x (Tensor): The input tensor representing raw values.
            x_mark (Tensor): The input tensor representing time-related features.

        Returns:
            Tensor: The combined embedding output after applying dropout.
        """
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_temp(nn.Module):
    """
    DataEmbedding is a neural network module for combining value, positional embeddings.

    This module is designed for sequential data processing, particularly when time-related features are
    important. It combines two different embeddings:
    1. Value Embedding: Transforms the raw values to a higher dimensional space.
    2. Positional Embedding: Adds information about the position of each element in the sequence.

    The output is a combination of these three embeddings, followed by dropout for regularization.

    Args:
        c_in (int): Number of input channels (features of the input data).
        d_model (int): Dimensionality of the embedding.
        dropout (float): Dropout rate for regularization. Default: 0.1.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super(DataEmbedding_wo_temp, self).__init__()

        # Initialize the various embeddings
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """
        Forward pass of the DataEmbedding layer.

        Args:
            x (Tensor): The input tensor representing raw values.
            x_mark (Tensor): The input tensor representing time-related features.

        Returns:
            Tensor: The combined embedding output after applying dropout.
        """
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    """
    DataEmbedding_wo_pos_temp is a neural network module for value embeddings.

    Value Embedding: Transforms the raw values to a higher dimensional space.

    Args:
        c_in (int): Number of input channels (features of the input data).
        d_model (int): Dimensionality of the embedding.
        dropout (float): Dropout rate for regularization. Default: 0.1.
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1) -> None:
        super(DataEmbedding_wo_pos_temp, self).__init__()

        # Initialize the various embeddings
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor, x_mark: Tensor) -> Tensor:
        """
        Forward pass of the DataEmbedding layer.

        Args:
            x (Tensor): The input tensor representing raw values.
            x_mark (Tensor): The input tensor representing time-related features.

        Returns:
            Tensor: The combined embedding output after applying dropout.
        """
        x = self.value_embedding(x)
        return self.dropout(x)
