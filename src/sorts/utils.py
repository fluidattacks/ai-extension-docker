import os
from typing import (
    List,
    NamedTuple,
)
from typing_extensions import TypedDict


FILE_FEATURES = [
    "num_commits",
    "num_unique_authors",
    "file_age",
    "midnight_commits",
    "risky_commits",
    "seldom_contributors",
    "num_lines",
    "commit_frequency",
    "busy_file",
    "extension",
]


class GitMetrics(TypedDict):
    author_email: List[str]
    commit_hash: List[str]
    date_iso_format: List[str]
    stats: List[str]


class FileFeatures(NamedTuple):
    num_commits: int
    num_unique_authors: int
    file_age: int
    midnight_commits: int
    risky_commits: int
    seldom_contributors: int
    num_lines: int
    commit_frequency: float
    busy_file: int
    extension: str


def get_path(file_name: str) -> str:
	return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)
