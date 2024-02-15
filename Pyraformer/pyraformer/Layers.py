#!/usr/bin/env python
# coding: utf-8

"""Layers section for Pyraformer model."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import math
from typing import List, Tuple, Union

import torch
from pyraformer.embed import CustomEmbedding, DataEmbedding
from pyraformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward
from torch import nn


def get_mask(
    input_size: int, window_size: List[int], inner_size: int, device: torch.device
) -> Tuple[torch.Tensor, List[int]]:
    """
    Get the attention mask of PAM-Naive.

    This function generates an attention mask for use in a hierarchical attention mechanism.
    The mask has intra-scale and inter-scale components. The intra-scale mask allows attention
    within a certain window around each element, and the inter-scale mask allows attention to the
    corresponding positions in adjacent scales.

    Args:
        input_size (int): The size of the input layer.
        window_size (List[int]): A list containing the window sizes for each layer in the model.
        inner_size (int): The size of the window for intra-scale attention.
        device (torch.device): The device on which the tensors should be allocated (e.g., CPU or CUDA).

    Returns:
        Tuple[torch.Tensor, List[int]]:
            A binary mask tensor of shape (seq_length, seq_length) indicating allowed attentions.
            A list of integers indicating the size of each layer.
    """
    # Get the size of all layers
    all_size = [input_size]
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[
                layer_idx - 1
            ]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (
                    i - start + 1
                ) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    mask = (1 - mask).bool()

    return mask, all_size


def refer_points(
    all_sizes: List[int], window_size: List[int], device: torch.device
) -> torch.Tensor:
    """
    Gather feature indices from PAM's pyramid sequences.

    This function calculates the indices in the pyramid attention mechanism that correspond to
    each point in the input sequence. For each point in the input layer, it finds the corresponding
    points in the subsequent layers of the pyramid.

    Args:
        all_sizes (List[int]): A list containing the sizes of each layer in the model.
        window_size (List[int]): A list containing the window sizes for each layer in the model.
        device (torch.device): The device on which the tensors should be allocated (e.g., CPU or CUDA).

    Returns:
        torch.Tensor: A tensor of indices with shape (1, input_size, len(all_sizes), 1) representing
            the index in each layer of the pyramid for each input point.
    """
    input_size = all_sizes[0]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(
                inner_layer_idx // window_size[j - 1], all_sizes[j] - 1
            )
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)

    return indexes.long()


def get_subsequent_mask(
    input_size: int, window_size: List[int], predict_step: int, truncate: bool
) -> torch.Tensor:
    """
    Get causal attention mask for the decoder in the model.

    This function generates a subsequent mask to enforce causality in the decoder's attention mechanism.
    The mask ensures that during prediction, the model can only attend to previous and current positions.
    If truncate is True, the mask is created for a truncated sequence, otherwise, it accounts for the entire sequence.

    Args:
        input_size (int): The size of the input layer.
        window_size (List[int]): A list containing the window sizes for each layer in the model.
        predict_step (int): The number of prediction steps.
        truncate (bool): A flag indicating whether the sequence should be truncated.

    Returns:
        torch.Tensor: A binary mask tensor representing allowed attentions. The shape of the mask is
            (1, predict_step, input_size + predict_step) if truncated, otherwise it takes into account
            the entire hierarchical structure.
    """
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][: input_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = [input_size]
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][: all_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(
    input_size: int, window_size: int, stride: int, device: torch.device
) -> torch.Tensor:
    """
    Get the indices of the keys that a given query needs to attend to.

    This function constructs a mask that specifies which key indices each query index should attend to,
    based on a hierarchical structure with multiple layers. The attention is constrained within a window
    around each query index, and the size of this window can vary at different levels of the hierarchy.

    Args:
        input_size (int): The size of the input layer.
        window_size (int): The size of the attention window.
        stride (int): The stride size for down sampling in the hierarchy.
        device (torch.device): The device on which the tensors should be allocated (e.g., CPU or CUDA).

    Returns:
        torch.Tensor: A tensor of shape (full_length, max_attn), where full_length is the total length of
            the hierarchical structure, and max_attn is the maximum attention window size in the structure.
            Each row of the tensor contains indices that the corresponding query index should attend to, with
            -1 indicating positions outside the attention window.
    """
    # Calculate sizes for each hierarchical layer
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size + i, 0:window_size] = (
            input_size + i + torch.arange(window_size) - window_size // 2
        )
        mask[input_size + i, mask[input_size + i] < input_size] = -1
        mask[input_size + i, mask[input_size + i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size + i, window_size : (window_size + stride)] = (
                torch.arange(stride) + i * stride
            )
        else:
            mask[input_size + i, window_size : (window_size + second_last)] = (
                torch.arange(second_last) + i * stride
            )

        mask[input_size + i, -1] = i // stride + third_start
        mask[input_size + i, mask[input_size + i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start + i, 0:window_size] = (
            third_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[third_start + i, mask[third_start + i] < third_start] = -1
        mask[third_start + i, mask[third_start + i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start + i, window_size : (window_size + stride)] = (
                input_size + torch.arange(stride) + i * stride
            )
        else:
            mask[third_start + i, window_size : (window_size + third_last)] = (
                input_size + torch.arange(third_last) + i * stride
            )

        mask[third_start + i, -1] = i // stride + fourth_start
        mask[third_start + i, mask[third_start + i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start + i, 0:window_size] = (
            fourth_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[fourth_start + i, mask[fourth_start + i] < fourth_start] = -1
        mask[fourth_start + i, mask[fourth_start + i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start + i, window_size : (window_size + stride)] = (
                third_start + torch.arange(stride) + i * stride
            )
        else:
            mask[fourth_start + i, window_size : (window_size + fourth_last)] = (
                third_start + torch.arange(fourth_last) + i * stride
            )

    return mask


def get_k_q(q_k_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the indices of the queries that a given key can attend to.

    This function constructs a mask that specifies which query indices each key index should attend to.
    It inverts the relationship defined in the q_k_mask so that for every key, you can determine which
    queries are allowed to consider it. The function assumes that the q_k_mask is a square matrix where
    the length of queries and keys is the same.

    Args:
        q_k_mask (torch.Tensor): A tensor of shape (N, M) where N is the number of queries and M is the
            number of keys. The tensor contains indices indicating which keys each query should attend to.

    Returns:
        torch.Tensor: A tensor of the same shape as q_k_mask, containing indices indicating which queries
            can attend to each key. If a query cannot attend to a key, the corresponding position is marked with -1.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                # Find the query indices in q_k_mask that can attend to key i
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]

    return k_q_mask


class EncoderLayer(nn.Module):
    """
    Compose an Encoder Layer with two sublayers: a self-attention mechanism (either standard MultiHeadAttention
    or PyramidalAttention) and a position-wise feedforward network.

    Args:
        d_model (int): The number of expected features in the input (required).
        d_inner (int): The dimensionality of the feed-forward layer (required).
        n_head (int): The number of heads in the multiheadattention models (required).
        d_k (int): The dimension of the key in the multiheadattention models (required).
        d_v (int): The dimension of the value in the multiheadattention models (required).
        dropout (float, optional): Dropout value (default=0.1).
        normalize_before (bool, optional): Whether to use layer normalization before the first sublayer or after the
            second sublayer (default=True).
        use_tvm (bool, optional): Whether to use PyramidalAttention instead of standard MultiHeadAttention (default=False).
        q_k_mask (torch.Tensor, optional): The mask tensor for query-key attention in PyramidalAttention (default=None).
        k_q_mask (torch.Tensor, optional): The mask tensor for key-query attention in PyramidalAttention (default=None).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        normalize_before: bool = True,
        use_tvm: bool = False,
        q_k_mask: torch.Tensor = None,
        k_q_mask: torch.Tensor = None,
    ):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention

            self.slf_attn = PyramidalAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                normalize_before=normalize_before,
                q_k_mask=q_k_mask,
                k_q_mask=k_q_mask,
            )
        else:
            self.slf_attn = MultiHeadAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                normalize_before=normalize_before,
            )

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(
        self, enc_input: torch.Tensor, slf_attn_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass the input through the encoder layer.

        Args:
            enc_input (torch.Tensor): The sequence to the encoder layer (required).
            slf_attn_mask (torch.Tensor, optional): The mask for the attention mechanism (default=None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The output of the encoder layer.
                The attention weights of the encoder layer.
        """
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask
            )

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """
    Compose a Decoder Layer with two sublayers: a self-attention mechanism (MultiHeadAttention)
    and a position-wise feedforward network.

    Args:
        d_model (int): The number of expected features in the input (required).
        d_inner (int): The dimensionality of the feed-forward layer (required).
        n_head (int): The number of heads in the multiheadattention models (required).
        d_k (int): The dimension of the key in the multiheadattention models (required).
        d_v (int): The dimension of the value in the multiheadattention models (required).
        dropout (float, optional): Dropout value (default=0.1).
        normalize_before (bool, optional): Whether to use layer normalization before the first sublayer or after the
            second sublayer (default=True).
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        n_head: int,
        d_k: int,
        d_v: int,
        dropout: float = 0.1,
        normalize_before: bool = True,
    ):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        slf_attn_mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pass the input (query, key, and value) through the decoder layer.

        Args:
            Q (torch.Tensor): The query tensor for the attention mechanism (required).
            K (torch.Tensor): The key tensor for the attention mechanism (required).
            V (torch.Tensor): The value tensor for the attention mechanism (required).
            slf_attn_mask (torch.Tensor, optional): The mask for the attention mechanism (default=None).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                The output of the decoder layer.
                The attention weights of the decoder layer.
        """
        enc_output, enc_slf_attn = self.slf_attn(Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    """
    A Convolutional Layer with downsampling followed by batch normalization and an ELU activation function.

    The layer performs a 1D convolution operation which effectively downsamples the input by using a stride
    equal to the kernel size. This is then followed by batch normalization and an ELU activation function.

    Args:
        c_in (int): The number of channels in the input (required).
        window_size (int): The size of the convolutional kernel, also the stride for downsampling (required).
    """

    def __init__(self, c_in: int, window_size: int):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=window_size,
            stride=window_size,
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the ConvLayer.

        Args:
            x (torch.Tensor): The input tensor to the ConvLayer (required).

        Returns:
            torch.Tensor: The output tensor after applying convolution, batch normalization, and ELU activation.
        """
        x = self.downConv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvConstruct(nn.Module):
    """
    Convolutional Construction for Sequence to Sequence Models (CSCM).

    This module constructs a series of convolutional layers for processing sequences. It is capable of
    handling variable window sizes for each convolutional layer. The output of each convolutional layer
    is concatenated and then normalized using Layer Normalization.

    Args:
        d_model (int): The number of expected features in the input (required).
        window_size (int or List[int]): The size(s) of the convolutional kernel(s). If a single integer is provided,
            all convolutional layers will use the same window size. If a list is provided, each convolutional layer will
            use the corresponding window size (required).
        d_inner (int): The dimensionality of the feed-forward layer within the convolutional layers (required).
    """

    def __init__(self, d_model: int, window_size: [int, list], d_inner: int):
        super(ConvConstruct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size[0]),
                    ConvLayer(d_model, window_size[1]),
                    ConvLayer(d_model, window_size[2]),
                ]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the ConvConstruct module.

        Args:
            enc_input (torch.Tensor): The input tensor to the ConvConstruct module (required).

        Returns:
            torch.Tensor: The normalized output tensor after applying a series of convolutional layers
                and concatenating their outputs.
        """
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class BottleneckConstruct(nn.Module):
    """
    Bottleneck Convolutional Construction for Sequence to Sequence Models (CSCM).

    This module applies a bottleneck approach to the input sequence, reducing its dimensionality before
    applying a series of convolutional layers and then projecting the output back to the original dimensionality.
    The output of each convolutional layer is concatenated and then normalized using Layer Normalization.

    Args:
        d_model (int): The number of expected features in the input (required).
        window_size (int or List[int]): The size(s) of the convolutional kernel(s). If a single integer is provided,
            all convolutional layers will use the same window size. If a list is provided, each convolutional layer will
            use the corresponding window size (required).
        d_inner (int): The dimensionality of the feed-forward layer within the convolutional layers (required).
    """

    def __init__(self, d_model: int, window_size: [int, list], d_inner: int):
        super(BottleneckConstruct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [ConvLayer(d_inner, ws) for ws in window_size]
            )
        self.up = nn.Linear(d_inner, d_model)
        self.down = nn.Linear(d_model, d_inner)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the Bottleneck_Construct module.

        Args:
            enc_input (torch.Tensor): The input tensor to the Bottleneck_Construct module (required).

        Returns:
            torch.Tensor: The normalized output tensor after applying a bottleneck transformation, a series
                of convolutional layers, and concatenating the outputs.
        """
        temp_input = self.down(enc_input).permute(0, 2, 1)
        all_inputs = []
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](temp_input)
            all_inputs.append(temp_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.up(all_inputs)
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)

        all_inputs = self.norm(all_inputs)

        return all_inputs


class MaxPoolingConstruct(nn.Module):
    """
    A neural network module implementing max pooling with configurable window sizes.

    This module applies max pooling to an input tensor using either a single window size
    for all pooling layers or unique window sizes for each layer. It also includes a
    layer normalization step after pooling.

    Args:
        d_model (int): The dimension of the model.
        window_size (Union[int, List[int]]): The size of the window for max pooling.
            If an integer is provided, all pooling layers use this size. If a list is
            provided, each pooling layer uses the corresponding size from the list.
        d_inner (int): Inner dimension parameter, currently unused.
    """

    def __init__(self, d_model: int, window_size: Union[int, List[int]], d_inner: int):
        super(MaxPoolingConstruct, self).__init__()
        # Initialize pooling layers based on the type of window_size provided
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [nn.MaxPool1d(kernel_size=window_size) for _ in range(3)]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [nn.MaxPool1d(kernel_size=size) for size in window_size]
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MaxPooling_Construct module.

        Applies max pooling and layer normalization to the input tensor.

        Args:
            enc_input (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output tensor after applying max pooling and normalization.
        """
        all_inputs = [enc_input.transpose(1, 2).contiguous()]

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        return self.norm(all_inputs)


class AvgPoolingConstruct(nn.Module):
    """
    A neural network module implementing average pooling with configurable window sizes.

    This module applies average pooling to an input tensor using either a single window size
    for all pooling layers or unique window sizes for each layer. It also includes a
    layer normalization step after pooling.

    Args:
        d_model (int): The dimension of the model.
        window_size (Union[int, List[int]]): The size of the window for average pooling.
            If an integer is provided, all pooling layers use this size. If a list is
            provided, each pooling layer uses the corresponding size from the list.
        d_inner (int): Inner dimension parameter, currently unused.
    """

    def __init__(self, d_model: int, window_size: Union[int, List[int]], d_inner: int):
        super(AvgPoolingConstruct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [
                    nn.AvgPool1d(kernel_size=window_size),
                    nn.AvgPool1d(kernel_size=window_size),
                    nn.AvgPool1d(kernel_size=window_size),
                ]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [
                    nn.AvgPool1d(kernel_size=window_size[0]),
                    nn.AvgPool1d(kernel_size=window_size[1]),
                    nn.AvgPool1d(kernel_size=window_size[2]),
                ]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the AvgPooling_Construct module.

        Applies average pooling and layer normalization to the input tensor.

        Args:
            enc_input (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output tensor after applying average pooling and normalization.
        """
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        all_inputs.append(enc_input)

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):
    """
    A neural network module for prediction tasks, using a linear transformation.

    This module is designed to perform a prediction task by applying a linear layer
    to the input data. The linear layer maps the input data to a specified number
    of output types. The weights of the linear layer are initialized using the
    Xavier normal initialization.

    Args:
        dim (int): The size of each input sample.
        num_types (int): The number of output types (i.e., the size of each output sample).
    """

    def __init__(self, dim: int, num_types: int):
        super(Predictor, self).__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Predictor module.

        Applies a linear transformation to the input data.

        Args:
            data (torch.Tensor): The input tensor to the module.

        Returns:
            torch.Tensor: The output tensor after applying the linear transformation.
        """
        out = self.linear(data)
        return out


class Decoder(nn.Module):
    """
    A decoder model with a self-attention mechanism.

    This class represents a decoder module typically used in sequence-to-sequence models. It includes
    multiple layers of the decoder, each with self-attention and potentially other mechanisms. The
    decoder is configurable for different embedding types and supports masking.

    Args:
        opt (Namespace): A configuration object with attributes such as d_model, d_inner_hid, n_head,
            d_k, d_v, dropout, embed_type, enc_in, covariate_size, seq_num.
        mask (Tensor): The mask tensor for the decoder.
    """

    def __init__(self, opt, mask: torch.Tensor):
        super(Decoder).__init__()

        self.model_type = opt.model
        self.mask = mask

        # Initialize decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    opt.d_k,
                    opt.d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                ),
                DecoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    opt.d_k,
                    opt.d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                ),
            ]
        )

        # Initialize embedding layer based on the specified type
        if opt.embed_type == "CustomEmbedding":
            self.dec_embedding = CustomEmbedding(
                opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout
            )
        else:
            self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(
        self, x_dec: torch.Tensor, x_mark_dec: torch.Tensor, refer: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the Decoder module.

        Processes the input through the embedding layer and subsequent decoder layers with self-attention.

        Args:
            x_dec (torch.Tensor): The input tensor for decoding.
            x_mark_dec (torch.Tensor): Additional tensor for marking or positional information in decoding.
            refer (torch.Tensor): Reference tensor for self-attention.

        Returns:
            torch.Tensor: The output tensor after decoding.
        """
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)

        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)

        return dec_enc
