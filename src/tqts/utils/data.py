#!/usr/bin/env python
# coding: utf-8

"""Data preprocessing module to convert text into tokens and create batches for training."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"


import torch
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path: str, seq_length: int = 512):
        """
        Dataset class for text data.

        Args:
            file_path (str): Path to the text file.
            seq_length (int, optional): Sequence length. Defaults to 512.
        """
        self.file_path = file_path
        self.seq_length = seq_length
        with open(file_path, "r", encoding="utf-8") as f:
            self.text_len = len(f.read())

        self.positions = list(range(self.text_len - self.seq_length))
        self.vocab = self._build_vocab()
        self.char2idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx2char = {idx: char for idx, char in enumerate(self.vocab)}

    def _build_vocab(self):
        """Build the vocabulary from the text file."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()
        vocab = sorted(set(text))
        return vocab

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        position = self.positions[idx]
        sequence = self._get_sequence(position)
        input_sequence = torch.tensor(
            [self.char2idx[ch] for ch in sequence[:-1]], dtype=torch.long
        )
        target_sequence = torch.tensor(
            [self.char2idx[ch] for ch in sequence[1:]], dtype=torch.long
        )
        return input_sequence, target_sequence

    def _get_sequence(self, position):
        """Get the sequence of characters at the given position."""
        with open(self.file_path, "r", encoding="utf-8") as f:
            f.seek(position)
            sequence = f.read(self.seq_length + 1)  # +1 for the target
        return sequence


def create_dataloader(file_path: str, seq_len: int, batch_size: int):
    dataset = TextDataset(file_path, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader, dataset.vocab, dataset.char2idx, dataset.idx2char
