import re
from typing import (
    Dict
)

COMMIT_RISK_LIMIT: int = 75

RENAME_REGEX: re.Pattern = re.compile(
    r"(?P<pre_path>.*)?"
    r"{(?P<old_name>.*) => (?P<new_name>.*)}"
    r"(?P<post_path>.*)?"
)

FEATURES_DICTS: Dict[str, str] = {
    "num_commits": "CM",
    "num_unique_authors": "AU",
    "file_age": "FA",
    "midnight_commits": "MC",
    "risky_commits": "RC",
    "seldom_contributors": "SC",
    "num_lines": "LC",
    "busy_file": "BF",
    "commit_frequency": "CF",
}

FEATURES_DESCRIPTION: Dict[str, str] = {
    "num_commits": "CM: Number of commits that the file has had since its creation.",
    "num_unique_authors": "AU: Number of unique authors that have modified the file.",
    "file_age": "FA: How many days have passed since the file was created.",
    "midnight_commits": "MC: Number of commits that were made between 12AM and 6AM.",
    "risky_commits": "RC: Number of commits that had over 200 deltas.",
    "seldom_contributors": "SC: The number of authors that have modified the file but rarely contribute to the repository.",
    "num_lines": "LC: The total number of lines that the file has.",
    "busy_file": "BF: The frequency with which the file is modified.",
    "commit_frequency": "CF: Wether the file has been modified by many authors since its creation.",
}
