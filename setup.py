#!/usr/bin/env python
# coding: utf-8

"""tqts package setup file."""

from pathlib import Path
import setuptools


__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

__version__ = "1.0"
"""Auto generated version number (use 'cz bump' to increment version number)."""

# insert requirements in setup.py for usage and develop
root_path = Path(__file__).parent

with open(root_path / "requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="tqts",
    version=__version__,
    author=__author__,
    author_email=__mail__,
    description="A Python package for time series quantization and forecasting.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dhanunjaya-Elluri/master-thesis",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
