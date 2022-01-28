import argparse
import base64
from file import (
    extract_features,
    get_extensions_list,
)
import git
from git.cmd import (
    Git,
)
from joblib import (
    load,
)
import numpy as np
from numpy import (
    ndarray,
)
import os
import pandas as pd
from pandas import (
    DataFrame,
)
from prettytable import (
    from_csv,
    PrettyTable,
)
import requests
import subprocess
import sys
from typing import (
    List,
    Tuple,
)
import urllib.request
from exceptions import (
    CredentialsError,
    CommitRiskError,
)
from utils import (
    get_path,
)
from constants import (
    COMMIT_RISK_LIMIT,
)

USERPASS = sys.argv[1]


def http_get(url: str, return_json: bool = True) -> str:
    b64 = base64.b64encode(USERPASS.encode()).decode()
    headers = {"Authorization": "Basic %s" % b64}
    response = requests.get(url=url, headers=headers)
    if response.status_code != 200:
        raise CredentialsError()

    return response.json() if return_json else response.text


def get_commit_files(url: str, file_paths: List[str]) -> None:
    for path in file_paths:
        url = url.replace("$path", path)
        response = http_get(url, return_json=False)
        file_name = path.split("/")[-1]
        with open(file_name, "w") as file:
            file.write(response)


def get_commit_files_paths(url) -> List[str]:
    response = http_get(url)
    return response["changes"]


def get_repositories_log(repo_path: str) -> None:
    git_repo: Git = git.Git(repo_path)
    git_log: str = git_repo.log(
        "--no-merges", "--numstat", "--pretty=%n%H,%ae,%aI%n"
    ).replace("\n\n\n", "\n")
    with open(
        f"{os.path.basename(repo_path)}.log", "w", encoding="utf8"
    ) as log_file:
        log_file.write(git_log)


def read_allowed_names() -> Tuple[List[str], ...]:
    allowed_names: List[List[str]] = []
    for name in ["extensions.lst", "composites.lst"]:
        with open(get_path(f"res/{name}")) as file:
            content_as_list = file.read().split("\n")
            allowed_names.append(list(filter(None, content_as_list)))

    return (allowed_names[0], allowed_names[1])


def get_subscription_files_df(repository_path: str) -> DataFrame:
    files: List[str] = []
    extensions, composites = read_allowed_names()
    ignore_dirs: List[str] = [".git"]
    repo_files = [
        os.path.join(path, filename).replace(
            f"{os.path.dirname(repository_path)}/", ""
        )
        for path, _, files in os.walk(repository_path)
        for filename in files
        if all([dir_ not in path for dir_ in ignore_dirs])
    ]

    allowed_files = list(
        filter(
            lambda ext: (
                ext in composites or ext.split(".")[-1].lower() in extensions
            ),
            repo_files,
        )
    )

    files_df: DataFrame = pd.DataFrame(allowed_files, columns=["file"])
    files_df["repo"] = repository_path
    files_df["is_vuln"] = 0

    return files_df


def build_results_csv(
    predictions: ndarray, predict_df: DataFrame, csv_name: str
) -> float:
    scope: str = csv_name.split(".")[0].split("_")[-1]
    result_df: DataFrame = pd.concat(
        [
            predict_df[[scope]],
            pd.DataFrame(
                predictions, columns=["pred", "prob_safe", "prob_vuln"]
            ),
        ],
        axis=1,
    )
    error: float = 5 + 5 * np.random.rand(
        len(result_df),
    )
    result_df["prob_vuln"] = round(result_df.prob_vuln * 100 - error, 1)
    sorted_files: DataFrame = (
        result_df[result_df.prob_vuln >= 0]
        .sort_values(by="prob_vuln", ascending=False)
        .reset_index(drop=True)[[scope, "prob_vuln"]]
    )
    sorted_files["file"] = sorted_files["file"].apply(
        lambda item: "/".join(item.split("/")[1:])
    )
    sorted_files["prob_vuln"] = sorted_files["prob_vuln"].apply(
        lambda item: f"{item}%"
    )
    sorted_files.to_csv(csv_name, index=False)

    prob_vulns = sorted_files["prob_vuln"].tolist()

    prob_vulns_mean = sum([float(prob.replace("%", "")) for prob in prob_vulns]) / len(prob_vulns)

    return prob_vulns_mean


def predict_vuln_prob(
    predict_df: DataFrame, features: List[str], csv_name: str
) -> float:
    model = load(get_path("res/model.joblib"))
    input_data = predict_df[model.feature_names + features]
    probability_prediction: ndarray = model.predict_proba(input_data)
    class_prediction: ndarray = model.predict(input_data)
    merged_predictions: ndarray = np.column_stack(
        [class_prediction, probability_prediction]
    )

    return build_results_csv(merged_predictions, predict_df, csv_name)


def display_results(csv_name: str) -> None:
    scope: str = csv_name.split(".")[0].split("_")[-1]
    with open(csv_name, "r", encoding="utf8") as csv_file:
        table = from_csv(
            csv_file, field_names=[scope, "prob_vuln"], delimiter=","
        )
    table.align[scope] = "l"
    table._max_width = {scope: 120, "prob_vuln": 10}

    print(table.get_string(start=1, end=20))


def prepare_sorts(
    repository_url: str, commit_file_paths: List[str]
) -> DataFrame:
    """Things to do before executing Sorts"""
    get_repositories_log(repository_url)
    files_df = get_subscription_files_df(repository_url)
    commit_files_rows = []
    for _, row in files_df.iterrows():
        if any(
            commit_file_path in row["file"]
            for commit_file_path in commit_file_paths
        ):
            commit_files_rows.append(row)

    commit_files_df = pd.DataFrame(commit_files_rows)
    extract_features(commit_files_df)

    return commit_files_df


def display_mean_risk(commit_mean_risk: int, commit_risk_limit: int) -> None:
    symbols = ["<", ">"]
    print(
        f"Mean Risk: {commit_mean_risk} "
        f"({symbols[commit_mean_risk > commit_risk_limit]} {commit_risk_limit} (limit))"
    )


def execute_sorts(files_df: DataFrame, break_pipeline: bool, commit_risk_limit: int) -> None:
    print("Sorts results")
    if not files_df.empty:
        results_file_name = "sorts_results_file.csv"
        extensions: List[str] = get_extensions_list()
        num_bits: int = len(extensions).bit_length()

        commit_mean_risk = predict_vuln_prob(
            files_df,
            [f"extension_{num}" for num in range(num_bits + 1)],
            results_file_name,
        )
        display_results(results_file_name)
        display_mean_risk(commit_mean_risk, commit_risk_limit)
        if break_pipeline and commit_mean_risk >= commit_risk_limit:
            raise CommitRiskError()
    else:
        print("No files in current commit: dataframe is empty")


def get_files(
    organization: str, project_name: str, repository_id: str, commit_id: str
):
    """Saves commit files locally and return its paths"""
    base_api_path = f"https://dev.azure.com/{organization}/{project_name}/_apis/git/repositories"
    commit_info_url = f"{base_api_path}/{repository_id}/commits/{commit_id}/changes?api-version=6.1-preview.1"
    items = get_commit_files_paths(commit_info_url)
    paths = [item["item"]["path"] for item in items if not "isFolder" in item["item"]]
    commit_files_url = f"{base_api_path}/{repository_id}/items?scopePath=$path&api-version=6.1-preview.1"
    get_commit_files(commit_files_url, paths)

    return paths

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("userpass", type=str)
    parser.add_argument("organization", type=str)
    parser.add_argument("project_name", type=str)
    parser.add_argument("repository_id", type=str)
    parser.add_argument("commit_id", type=str)
    parser.add_argument("repo_local_url", type=str)
    parser.add_argument("break_pipeline", type=str)
    parser.add_argument("commit_risk_limit", type=int, default=COMMIT_RISK_LIMIT)

    return parser.parse_args()


def main():
    args = get_args()

    # Get commit files
    commit_file_paths = get_files(
        args.organization, args.project_name, args.repository_id, args.commit_id
    )

    # Prepare Sorts
    files_df = prepare_sorts(args.repo_local_url, commit_file_paths)

    # Execute Sorts
    execute_sorts(files_df, args.break_pipeline, args.commit_risk_limit)


if __name__ == "__main__":
    main()
