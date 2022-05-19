import os
import pandas as pd
from pandas import (
    DataFrame,
)
from typing import(
     List,
     NamedTuple,
     Tuple,
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
    num_commits: List[int]
    num_unique_authors: List[int]
    file_age: List[int]
    midnight_commits: List[int]
    risky_commits: List[int]
    seldom_contributors: List[int]
    num_lines: List[int]
    commit_frequency: List[float]
    busy_file: List[int]
    extension: str


def get_path(file_name: str) -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name)


def split_training_data(
    training_df: DataFrame, feature_list: Tuple[str, ...]
) -> Tuple[DataFrame, DataFrame]:
    # Separate the labels from the features in the training data
    filtered_df = pd.concat(
        [
            # Include labels
            training_df.iloc[:, 0],
            # Include features
            training_df.loc[:, feature_list],
            # Include all extensions
            training_df.loc[
                :, training_df.columns.str.startswith("extension_")
            ],
        ],
        axis=1,
    )
    filtered_df.dropna(inplace=True)

    return filtered_df.iloc[:, 1:], filtered_df.iloc[:, 0]
