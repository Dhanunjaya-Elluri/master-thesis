#!/usr/bin/env python
# coding: utf-8

"""Masking layers for Informer."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import math

import numpy as np
import torch


class LocalMask:
    def __init__(self, B: int, L: int, S: int, device: str = "cpu"):
        """A class for creating a local mask tensor used in attention mechanisms.

        This mask is designed to allow selective attention within a specified local range.

        Args:
            B (int): The batch size.
            L (int): The sequence length.
            S (int): The size of the second dimension of the mask (typically the same as L).
            device (str): The device to which the mask tensor is to be sent. Defaults to "cpu".
        """
        mask_shape = [B, 1, L, S]
        with torch.no_grad():
            self.len = math.ceil(np.log2(L))
            self._mask1 = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)
            self._mask2 = ~torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=-self.len
            ).to(device)
            self._mask = self._mask1 + self._mask2

    @property
    def mask(self) -> torch.Tensor:
        """A property to access the generated local mask tensor.

        Returns:
            torch.Tensor: The local mask tensor.
        """
        return self._mask


class TriangularCasualMask:
    """Triangular Casual Mask module for the Informer Model."""

    def __init__(self, B: int, L: int, device: str = "cpu"):
        """Initialize the Triangular Casual Mask module.

        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (torch.device): Device to use.
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        """Get the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (B, 1, L, L).
        """
        return self._mask


class ProbMask:
    def __init__(
        self,
        B: int,
        H: int,
        L: int,
        index: int,
        scores: torch.Tensor,
        device: str = "cpu",
    ):
        """Initialize the Prob Mask module.

        Args:
            B (int): Batch size.
            H (int): Number of heads.
            L (int): Sequence length.
            index (int): Index of the mask.
            scores (torch.Tensor): Scores tensor of shape (B, H, L, L).
            device (torch.device): Device to use.
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[
            torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :
        ].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """Get the mask tensor.

        Returns:
            torch.Tensor: Mask tensor of shape (B, H, L, L).
        """
        return self._mask
