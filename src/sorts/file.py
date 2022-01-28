from category_encoders import (
    BinaryEncoder,
)
from cryptography.fernet import (
    Fernet,
)
from datetime import (
    datetime,
)
import dateutil
from functools import (
    partial,
)
import git
from git.cmd import (
    Git,
)
from git.exc import (
    GitCommandError,
    GitCommandNotFound,
)
from numpy import (
    ndarray,
)
import os
import pandas as pd
from pandas import (
    DataFrame,
    Series,
)
import pytz
import re
import tempfile
import time
from tqdm import (
    tqdm,
)
from typing import (
    List,
    NamedTuple,
    Set,
    Tuple,
)
from exceptions import (
    AIExtensionError,
)
from utils import (
    FILE_FEATURES,
    FileFeatures,
    get_path,
    GitMetrics,
)


def get_extensions_list() -> List[str]:
    extensions: List[str] = []
    with open(get_path("res/extensions.lst"), "r", encoding="utf8") as file:
        extensions = [line.rstrip() for line in file]

    return extensions


def get_log_file_metrics(logs_dir: str, repo: str, file: str) -> GitMetrics:
    git_metrics: GitMetrics = GitMetrics(
        author_email=[], commit_hash=[], date_iso_format=[], stats=[]
    )
    cursor: str = ""
    with open(f"{repo}.log", "r", encoding="utf8") as log_file:
        for line in log_file:
            # An empty line marks the start of a new commit diff
            if not line.strip("\n"):
                cursor = "info"
                continue
            # Next, there is a line with the format 'Hash,Author,Date'
            if cursor == "info":
                info: List[str] = line.strip("\n").split(",")
                commit: str = info[0]
                author: str = info[1]
                date: str = info[2]
                cursor = "diff"
                continue
            # Next, there is a list of changed files and the changed lines
            # with the format 'Additions    Deletions   File'
            if cursor == "diff":
                changed_name: bool = False
                # Keeps track of the file if its name was changed
                if "=>" in line:
                    match = re.match(
                        RENAME_REGEX, line.strip("\n").split("\t")[2]
                    )
                    if match:
                        path_info: Dict[str, str] = match.groupdict()
                        if file == (
                            f'{path_info["pre_path"]}{path_info["new_name"]}'
                            f'{path_info["post_path"]}'
                        ):
                            changed_name = True
                            file = (
                                f'{path_info["pre_path"]}'
                                f'{path_info["old_name"]}'
                                f'{path_info["post_path"]}'
                            )
                if file in line or changed_name:
                    git_metrics["author_email"].append(author)
                    git_metrics["commit_hash"].append(commit)
                    git_metrics["date_iso_format"].append(date)

                    stats: List[str] = line.strip("\n").split("\t")
                    git_metrics["stats"].append(
                        f"1 file changed, {stats[0]} insertions (+), "
                        f"{stats[1]} deletions (-)"
                    )

    return git_metrics


def get_features(row: Series, logs_dir: str) -> FileFeatures:
    # Use -1 as default value to avoid ZeroDivisionError
    file_age: int = -1
    midnight_commits: int = -1
    num_commits: int = -1
    num_lines: int = -1
    risky_commits: int = -1
    seldom_contributors: int = -1
    unique_authors: List[str] = []
    extension: str = ""

    try:
        repo_path: str = row["repo"]
        repo_name: str = os.path.basename(repo_path)
        file_relative: str = row["file"].replace(f"{repo_name}/", "", 1)
        git_metrics: GitMetrics = get_log_file_metrics(
            logs_dir, repo_name, file_relative
        )
        file_age = get_file_age(git_metrics)
        midnight_commits = get_midnight_commits(git_metrics)
        num_commits = get_num_commits(git_metrics)
        num_lines = get_num_lines(os.path.join(repo_path, file_relative))
        risky_commits = get_risky_commits(git_metrics)
        seldom_contributors = get_seldom_contributors(git_metrics)
        unique_authors = get_unique_authors(git_metrics)
        extension = file_relative.split(".")[-1].lower()
    except (FileNotFoundError, IndexError) as exception:
        raise AIExtensionError(f"get_features: {exception}")

    return FileFeatures(
        num_commits=num_commits,
        num_unique_authors=len(unique_authors),
        file_age=file_age,
        midnight_commits=midnight_commits,
        risky_commits=risky_commits,
        seldom_contributors=seldom_contributors,
        num_lines=num_lines,
        commit_frequency=(
            round(num_commits / file_age, 4) if file_age else num_commits
        ),
        busy_file=1 if len(unique_authors) > 9 else 0,
        extension=extension,
    )


def get_file_age(git_metrics: GitMetrics) -> int:
    today: datetime = datetime.now(pytz.utc)
    commit_date_history: List[str] = git_metrics["date_iso_format"]
    file_creation_date: str = commit_date_history[-1]

    return (today - dateutil.parser.isoparse(file_creation_date)).days


def get_num_commits(git_metrics: GitMetrics) -> int:
    commit_history: List[str] = git_metrics["commit_hash"]

    return len(commit_history)


def get_num_lines(file_path: str) -> int:
    result: int = 0
    try:
        with open(file_path, "rb") as file:
            bufgen = iter(partial(file.raw.read, 1024 * 1024), b"")  # type: ignore
            result = sum(buf.count(b"\n") for buf in bufgen)
    except FileNotFoundError as exception:
        raise AIExtensionError(f"get_num_lines: {exception}")

    return result


def get_midnight_commits(git_metrics: GitMetrics) -> int:
    commit_date_history: List[str] = git_metrics["date_iso_format"]
    commit_hour_history: List[int] = [
        dateutil.parser.isoparse(date).hour for date in commit_date_history
    ]

    return sum([1 for hour in commit_hour_history if 0 <= hour < 6])


def get_risky_commits(git_metrics: GitMetrics) -> int:
    risky_commits: int = 0
    commit_stat_history: List[str] = git_metrics["stats"]
    for stat in commit_stat_history:
        insertions, deletions = parse_git_shortstat(stat.replace("--", ", "))
        if insertions + deletions > 200:
            risky_commits += 1

    return risky_commits


def get_seldom_contributors(git_metrics: GitMetrics) -> int:
    seldom_contributors: int = 0
    authors_history: List[str] = git_metrics["author_email"]
    unique_authors: Set[str] = set(authors_history)
    avg_commit_per_author: float = round(
        len(authors_history) / len(unique_authors), 4
    )
    for author in unique_authors:
        commits: int = authors_history.count(author)
        if commits < avg_commit_per_author:
            seldom_contributors += 1

    return seldom_contributors


def get_unique_authors(git_metrics: GitMetrics) -> List[str]:
    authors_history: List[str] = list(set(git_metrics["author_email"]))
    authors_history_names: List[str] = [
        author.split("@")[0] for author in authors_history
    ]
    for index, author_name in enumerate(authors_history_names):
        if authors_history_names.count(author_name) > 1:
            del authors_history[index]
            del authors_history_names[index]

    return authors_history


def encrypt_column_values(value: str) -> str:
    fernet = Fernet(Fernet.generate_key())

    return fernet.encrypt(value.encode()).decode()


def parse_git_shortstat(stat: str) -> Tuple[int, int]:
    stat_regex: re.Pattern = re.compile(
        r"([0-9]+ files? changed)?"
        r"(, (?P<insertions>[0-9]+) insertions\(\+\))?"
        r"(, (?P<deletions>[0-9]+) deletions\(\-\))?"
    )
    insertions: int = 0
    deletions: int = 0
    match = re.match(stat_regex, stat.strip())
    if match:
        groups: Dict[str, str] = match.groupdict()
        if groups["insertions"]:
            insertions = int(groups["insertions"])
        if groups["deletions"]:
            deletions = int(groups["deletions"])

    return insertions, deletions


def encode_extensions(training_df: DataFrame) -> None:
    extensions: List[str] = get_extensions_list()
    extensions_df: DataFrame = pd.DataFrame(extensions, columns=["extension"])
    encoder: BinaryEncoder = BinaryEncoder(cols=["extension"], return_df=True)
    encoder.fit(extensions_df)
    encoded_extensions = encoder.transform(training_df[["extension"]])
    encoded_extensions.insert(0, "extension_10", 0)
    cols = encoded_extensions.columns.tolist()
    cols.insert(11, cols.pop(0))
    training_df[cols] = encoded_extensions.values.tolist()


def format_dataset(training_df: DataFrame) -> DataFrame:
    training_df.drop(
        training_df[training_df["file_age"] == -1].index, inplace=True
    )
    training_df.reset_index(inplace=True, drop=True)
    encode_extensions(training_df)

    return training_df


def get_repositories_log(dir_: str, repos_paths: ndarray) -> None:
    for repo_path in repos_paths:
        repo: str = os.path.basename(repo_path)
        git_repo: Git = git.Git(repo_path)
        git_log: str = git_repo.log(
            "--no-merges", "--numstat", "--pretty=%n%H,%ae,%aI%n"
        ).replace("\n\n\n", "\n")
        with open("repo.log", "w", encoding="utf8") as log_file:
            log_file.write(git_log)


def extract_features(training_df: DataFrame) -> bool:
    success: bool = True
    try:
        timer: float = time.time()
        with tempfile.TemporaryDirectory() as tmp_dir:
            get_repositories_log(tmp_dir, training_df["repo"].unique())
            tqdm.pandas()

            # Get features into dataset
            training_df[FILE_FEATURES] = training_df.progress_apply(
                get_features, args=(tmp_dir,), axis=1, result_type="expand"
            )
            format_dataset(training_df)
    except KeyError as exception:
        success = False
        raise AIExtensionError(f"extract_features: {exception}")

    return success
