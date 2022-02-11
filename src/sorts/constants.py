import re

COMMIT_RISK_LIMIT: int = 75

RENAME_REGEX: re.Pattern = re.compile(
    r"(?P<pre_path>.*)?"
    r"{(?P<old_name>.*) => (?P<new_name>.*)}"
    r"(?P<post_path>.*)?"
)
