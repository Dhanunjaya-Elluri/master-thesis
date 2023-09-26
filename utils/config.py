"""Utility file for loading yaml configuration files."""

import os
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../config/')


def load_config(file_name: str, path=CONFIG_PATH):
    """Load yaml configuration file.

    Args:
        file_name (str): Name of the yaml configuration file.
        path (str, optional): Path to the yaml configuration file. Defaults to CONFIG_PATH.

    Returns:
        dict: Dictionary containing the configuration.
        """
    with open(os.path.join(path, file_name), 'r') as f:
        config = yaml.safe_load(f)
    return config
