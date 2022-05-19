#!/usr/bin/env python3

import argparse
import os
from typing import (
    List,
    Tuple
)

import git
import numpy as np
import pandas as pd
import shap
from constants import (
    COMMIT_RISK_LIMIT,
    FEATURES_DESCRIPTION,
    FEATURES_DICTS,
)
from exceptions import (
    CommitRiskError,
)
from file import (
    extract_features,
    get_extensions_list,
)
from git.cmd import (
    Git,
)
from joblib import (
    load,
)
from numpy import (
    ndarray,
)
from pandas import (
    DataFrame,
)
from prettytable import (
    from_csv,
)
from utils import (
    get_path,
    split_training_data,
)

MODEL = load(get_path("res/model.joblib"))


def load_training_data() -> DataFrame:
    """Load a DataFrame with the training data in CSV format"""
    input_file: str = get_path("res/binary_encoded_training_data.csv")
    data: DataFrame = pd.read_csv(input_file, engine="python")

    return data


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


def format_prediction_results(
    predictions: ndarray, predict_df: DataFrame
) -> float:
    result_df: DataFrame = pd.concat(
        [
            predict_df[["file"]],
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
    result_df["prob_vuln"] = result_df["prob_vuln"].clip(lower=0)
    result_df: DataFrame = result_df.reset_index(drop=True)[
        ["file", "prob_vuln"]
    ]
    result_df["file"] = result_df["file"].apply(
        lambda item: "/".join(item.split("/")[1:])
    )

    return result_df


def get_merged_predictions(input_data: DataFrame) -> ndarray:
    probability_prediction: ndarray = MODEL.predict_proba(input_data)
    class_prediction: ndarray = MODEL.predict(input_data)
    merged_predictions: ndarray = np.column_stack(
        [class_prediction, probability_prediction]
    )
    return merged_predictions


def classify_shap_values(shap_values: List, expected_value: float) -> List:
    classified_values = [""] * len(shap_values)
    for i in range(len(shap_values)):
        if abs(shap_values[i]) > expected_value * 0.7:
            classified_values[i] = "Big"
        elif abs(shap_values[i]) > expected_value * 0.35:
            classified_values[i] = "Moderate"
        elif abs(shap_values[i]) > expected_value * 0.05:
            classified_values[i] = "Small"
        else:
            classified_values[i] = "Negligible"
        if shap_values[i] >= 0:
            classified_values[i] = classified_values[i] + " Increase"
        else:
            classified_values[i] = classified_values[i] + " Decrease"

    return classified_values


def predict_vuln_prob(
    predict_df: DataFrame, extensions: List[str], csv_name: str
) -> List[float]:
    feature_names = MODEL.feature_names
    prev_input_data = predict_df[feature_names + extensions]
    current_input_data = predict_df[feature_names + extensions]
    feature_quantity = len(feature_names)

    # Separate previous and current commit data
    for feature in feature_names:
        prev_input_data[str(feature)] = prev_input_data[str(feature)].apply(
            lambda item: item[0]
        )
        current_input_data[str(feature)] = current_input_data[
            str(feature)
        ].apply(lambda item: item[1])

    prev_commit_results: DataFrame = format_prediction_results(
        get_merged_predictions(prev_input_data),
        predict_df,
    )
    current_commit_results: DataFrame = format_prediction_results(
        get_merged_predictions(current_input_data),
        predict_df,
    )

    prev_prob_vulns = prev_commit_results["prob_vuln"].tolist()
    current_prob_vulns = current_commit_results["prob_vuln"].tolist()
    prev_prob_vulns_mean = sum(prev_prob_vulns) / len(prev_prob_vulns)
    current_prob_vulns_mean = sum(current_prob_vulns) / len(current_prob_vulns)

    # Use shap to calculate the contribution of each feature to the risk
    training_data = load_training_data()
    X_train = split_training_data(training_data, feature_names)[0]
    X_train_sampled = shap.sample(X_train, 100)
    explainerModel = shap.KernelExplainer(MODEL.predict, X_train_sampled)
    shap_values_Model = explainerModel.shap_values(current_input_data)
    risky_features_indexes = [
        max(enumerate(shap_values[:feature_quantity]), key=lambda x: x[1])[0]
        for shap_values in shap_values_Model
    ]

    # Format and organize result data in a CSV
    current_commit_results["risk_delta"] = (
        current_commit_results["prob_vuln"] - prev_commit_results["prob_vuln"]
    )
    current_commit_results[feature_names] = [
        classify_shap_values(
            shap_values[:feature_quantity], explainerModel.expected_value
        )
        for shap_values in shap_values_Model
    ]
    current_commit_results["biggest_contributor"] = [
        FEATURES_DICTS[feature_names[i]] for i in risky_features_indexes
    ]
    current_commit_results = current_commit_results[
        current_commit_results.prob_vuln >= 0
    ].sort_values(by="prob_vuln", ascending=False)
    current_commit_results["risk_delta"] = current_commit_results[
        "risk_delta"
    ].apply(lambda item: f"{round(item, 2)}%")
    current_commit_results["prob_vuln"] = current_commit_results[
        "prob_vuln"
    ].apply(lambda item: f"{round(item, 2)}%")
    current_commit_results.to_csv(csv_name, index=False)

    return [prev_prob_vulns_mean, current_prob_vulns_mean]


def display_results(csv_name: str) -> None:
    feature_list = [
        FEATURES_DICTS[feature_name] for feature_name in MODEL.feature_names
    ]
    field_names = (
        ["File", "Current Risk", "Risk Increment"]
        + [s + " Risk Contribution" for s in feature_list]
        + ["Biggest Risk Contributor"]
    )
    with open(csv_name, "r", encoding="utf8") as csv_file:
        table = from_csv(csv_file, field_names=field_names, delimiter=",")

    table.align = "l"
    for feature in MODEL.feature_names:
        print(f"{FEATURES_DESCRIPTION[feature]}")
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


def display_mean_risk(
    commit_mean_risk: List[int], commit_risk_limit: int
) -> None:
    symbols = ["<", ">"]
    print(
        f"Previous mean Risk: {commit_mean_risk[0]} \n"
        f"Current mean Risk: {commit_mean_risk[1]} "
        f"({symbols[commit_mean_risk[1] > commit_risk_limit]} {commit_risk_limit} (limit))"
    )


def execute_sorts(
    files_df: DataFrame, break_pipeline: bool, commit_risk_limit: int
) -> None:
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
        if break_pipeline and commit_mean_risk[1] >= commit_risk_limit:
            raise CommitRiskError()
    else:
        print("There are no valid files in current commit: dataframe is empty")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_local_path", type=str)
    parser.add_argument("break_pipeline", type=str)
    parser.add_argument(
        "commit_risk_limit", type=int, default=COMMIT_RISK_LIMIT
    )

    return parser.parse_args()


def main():
    args = get_args()

    # Get commit files
    git_repo: Git = git.Git(args.repo_local_path)
    diff = git_repo.diff("HEAD~1..HEAD", name_only=True)
    commit_file_paths = diff.split("\n")

    # Prepare Sorts
    files_df = prepare_sorts(args.repo_local_path, commit_file_paths)

    # Execute Sorts
    execute_sorts(files_df, args.break_pipeline, args.commit_risk_limit)


if __name__ == "__main__":
    main()
