#!/usr/bin/env python
# coding: utf-8

"""Module to download and prepare timeseries datasets."""

__author__ = "Dhanunjaya Elluri"
__mail__ = "dhanunjayet@gmail.com"

import os
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass
class Dataset:
    name: str
    url: str
    path: str
    folder: Optional[str] = None

    def __post_init__(self):
        self.repo_name = self.url.split("/")[-1]
        self.user_name = self.url.split("/")[-2]
        self.data_file = f"{self.name}.csv"

    def file_exists(self) -> bool:
        """Check if the dataset is already downloaded."""
        return os.path.exists(os.path.join(self.path, self.data_file))

    def get_branch_name(self) -> str:
        """Get the branch name of the dataset."""
        url = f"https://api.github.com/repos/{self.user_name}/{self.repo_name}/branches"
        branches = requests.get(url).json()
        branch = None
        for b in branches:
            if b["name"] in ["master", "main"]:
                branch = b
                break
        if branch is None:
            raise ValueError(f"Could not find master/main branch for {self.name}")
        return branch["name"]

    def build_full_url(self, branch_name: str) -> str:
        """Build the full url of the dataset."""

        repo_url = self.url.replace(
            "https://github.com", "https://raw.githubusercontent.com"
        )
        if self.folder is not None:
            repo_url = f"{repo_url}/{branch_name}/{self.folder}"
        else:
            repo_url = f"{repo_url}/{branch_name}"
        full_url = f"{repo_url}/{self.data_file}"
        return full_url

    @staticmethod
    def download_and_save(url: str, path: str, name: str, data_file: str) -> None:
        """Download a file from a given url and save it to the given path."""
        if requests.get(url).status_code != 200:
            raise ValueError(f"Could not find {name} dataset at {url}")
        r = requests.get(url)
        # Download dataset as csv file to the given path
        with open(os.path.join(path, data_file), "wb") as f:
            f.write(r.content)

    def download(self) -> None:
        """Download the dataset."""
        # Check if the dataset is already downloaded.
        if self.file_exists():
            print(f"{self.name} dataset already exists at {self.path}")
            return

        branch = self.get_branch_name()

        full_url = self.build_full_url(branch)
        self.download_and_save(full_url, self.path, self.name, self.data_file)
        print(f"Downloaded {self.name} dataset to {self.path}")
