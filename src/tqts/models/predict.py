#!/usr/bin/env python
# coding: utf-8

"""Module to predict the list of characters from a given sequence."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjaya.elluri@tu-dortmund.de"

# import os
import argparse
import torch
import torch.nn as nn
from tqts.models.Transformer import Transformer
from tqts.utils.config import load_config
from tqts.utils.dataloader import CharDataset


# load saved models using .pt file
def load_model(model_path: str, device: str, config: dict) -> nn.Module:
    """Load the saved models.

    Args:
        config (dict): Config dictionary.
        model_path (str): Path to the saved models.
        device (str): Device to load the models to.

    Returns:
        nn.Module: Loaded models.
    """
    model = Transformer(
        vocab_size=config["models"]["vocab_size"],
        d_model=config["models"]["d_model"],
        num_heads=config["models"]["num_heads"],
        num_encoder_layers=config["models"]["num_encoder_layers"],
        num_decoder_layers=config["models"]["num_decoder_layers"],
        dim_feedforward=config["models"]["dim_feedforward"],
        dropout=config["models"]["dropout"],
        activation=config["models"]["activation"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def predict_next_char(
    model: nn.Module, text: str, char_to_int: dict, int_to_char: dict, d_model: int
) -> str:
    """Predict all characters given a sequence of characters."""

    # models.eval()
    input_seq = [char_to_int.get(ch, 0) for ch in text[-d_model:]]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0)
    target_seq = torch.ones_like(input_seq, dtype=torch.long)

    with torch.no_grad():
        output = model(input_seq)

    predicted_seq = ""
    for i in range(output.shape[1]):
        predicted_index = torch.argmax(output[0, i, :]).item()
        predicted_seq += int_to_char[predicted_index]

    return predicted_seq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/transformer.yaml",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/ETTh1_model.pt",
        help="Path to the saved models.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to load the models to."
    )
    parser.add_argument(
        "--text",
        type=str,
        default="",
        help="Text to predict the next character from.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model = load_model(args.model_path, args.device, config)
    dataset = CharDataset(config["data"]["file_path"], config["data"]["seq_len"])
    char_to_int = dataset.char_to_int
    int_to_char = dataset.int_to_char
    print(f"int to char: {int_to_char}")
    predicted_seq = predict_next_char(model, args.text, char_to_int, int_to_char, 128)
    print(f"Predicted seq: {predicted_seq}")


if __name__ == "__main__":
    main()
