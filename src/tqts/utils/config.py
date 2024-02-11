#!/usr/bin/env python
# coding: utf-8

"""Utility file for loading yaml configuration files."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import os

import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/")


def load_config(file_name: str):
    """Load yaml configuration file.

    Args:
        file_name (str): Name of the yaml configuration file.

    Returns:
        dict: Dictionary containing the configuration.
    """
    with open(file_name, "r") as f:
        config = yaml.safe_load(f)
    return config
