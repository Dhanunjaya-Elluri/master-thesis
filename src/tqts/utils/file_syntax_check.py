#!/usr/bin/env python
# coding: utf-8

""" File syntax check for JSON and YAML files. """

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import json
import os
from pathlib import Path
from typing import List, Tuple

import yaml


def check_file_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """
    Check the syntax of a JSON or YAML file.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating whether the file is valid and a list of errors.
    """

    file_extension = Path(file_path).suffix
    if file_extension == ".json":
        return check_json_file_syntax(file_path)
    elif file_extension == ".yaml" or file_extension == ".yml":
        return check_yaml_file_syntax(file_path)
    else:
        raise ValueError(
            "Invalid file extension. Supported extensions: '.json', '.yaml', '.yml'."
        )


def check_json_file_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """
    Check the syntax of a JSON file.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating whether the file is valid and a list of errors.
    """

    with open(file_path) as f:
        try:
            json.load(f)
            return True, []
        except json.JSONDecodeError as e:
            return False, [str(e)]


def check_yaml_file_syntax(file_path: str) -> Tuple[bool, List[str]]:
    """
    Check the syntax of a YAML file.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating whether the file is valid and a list of errors.
    """

    with open(file_path) as f:
        try:
            yaml.safe_load(f)
            return True, []
        except yaml.YAMLError as e:
            return False, [str(e)]


def check_file_syntax_recursive(directory: str) -> Tuple[bool, List[str]]:
    """
    Recursively check the syntax of all JSON and YAML files in a directory.

    Args:
        directory (str): The path to the directory to check.

    Returns:
        Tuple[bool, List[str]]: A tuple containing a boolean indicating whether the files are valid and a list of errors.
    """

    file_paths = get_file_paths_recursive(directory)
    errors = []
    for file_path in file_paths:
        valid, file_errors = check_file_syntax(file_path)
        if not valid:
            errors.extend(file_errors)
    return len(errors) == 0, errors


def get_file_paths_recursive(directory: str) -> List[str]:
    """
    Recursively get the paths of all JSON and YAML files in a directory.

    Args:
        directory (str): The path to the directory to check.

    Returns:
        List[str]: A list of file paths.
    """

    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if (
                file.endswith(".json")
                or file.endswith(".yaml")
                or file.endswith(".yml")
            ):
                file_paths.append(os.path.join(root, file))
    return file_paths
