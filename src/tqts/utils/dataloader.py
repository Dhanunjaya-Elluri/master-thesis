#!/usr/bin/env python
# coding: utf-8

"""Data preprocessing module to convert text into tokens and create batches for training."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):
    """CharDataset for Next Character Prediction task from a text file."""

    def __init__(self, file_path: str, seq_len: int) -> None:
        """Initialize the CharDataset.

        Args:
            file_path (str): Path to the text file.
            seq_len (int): Sequence length.
        """
        self.seq_len = seq_len
        text = self._load_text(file_path)
        self.chars, self.char_to_int, self.int_to_char = self._create_vocab(text)
        self.input_seq, self.target_seq = self._create_sequence(text)

    @staticmethod
    def _load_text(file_path) -> str:
        """Load the text file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            str: Text from the file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _create_vocab(text) -> tuple:
        """Create vocabulary from the text.

        Args:
            text (str): Text from the file.

        Returns:
            tuple: Tuple containing vocabulary and reverse vocabulary.
        """
        chars = sorted(list(set(text)))
        char_to_int = {ch: i for i, ch in enumerate(chars)}
        int_to_char = {i: ch for i, ch in enumerate(chars)}
        return chars, char_to_int, int_to_char

    def _create_sequence(self, text) -> tuple:
        """Create input and target sequences from the text.

        Args:
            text (str): Text from the file.

        Returns:
            tuple: Tuple containing input and target sequences.
        """
        input_seq = []
        target_seq = []
        for i in range(len(text) - self.seq_len):
            input_seq.append(text[i : i + self.seq_len])
            target_seq.append(text[i + 1 : i + self.seq_len + 1])
        return input_seq, target_seq

    def __len__(self):
        return len(self.input_seq)

    def __getitem__(self, idx):
        input_seq, target_seq = self.input_seq[idx], self.target_seq[idx]
        return (
            torch.tensor([self.char_to_int[ch] for ch in input_seq], dtype=torch.long),
            torch.tensor([self.char_to_int[ch] for ch in target_seq], dtype=torch.long),
        )


def collate_fn(batch: list) -> tuple:
    """Collate function for the DataLoader.

    Args:
        batch (list): List of tuples containing input and target sequences.

    Returns:
        tuple: Tuple containing input and target tensors.
    """
    input_seq, target_seq = zip(*batch)
    input_seq = torch.stack(input_seq, dim=1)
    target_seq = torch.stack(target_seq, dim=1)
    return input_seq, target_seq
