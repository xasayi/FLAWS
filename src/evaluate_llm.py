import os
import csv
import json
from collections import defaultdict
from src.pipeline.single_process.single_error_identification import (
    identify_and_evaluate,
)
from src.utils.formatting import format_claims


def compute_accuracy(identification_dict: dict[str, list]) -> float:
    total = 0
    correct = 0
    for _, entries in identification_dict.items():
        for identify in entries:
            correct += list(identify.values())[0]
            total += 1

    return correct / total if total > 0 else 0


def generate_csv_results(
    model_names: dict[str, list[str]],
    dataset_name: str,
    model_identification: str,
) -> None:
    all_results = []

    for model_insertion, name in model_names.items():
        version_control = name[0]
        file = f"{dataset_name}/{version_control}/{name[1]}.json"

        os.makedirs(f"{dataset_name}/{version_control}", exist_ok=True)
        os.makedirs(f"{dataset_name}/{version_control}/altered_papers", exist_ok=True)

        claim_folder = f"{dataset_name}/{version_control}/generated_claims"
        error_folder = f"{dataset_name}/{version_control}/inserted_error"
        location_folder = f"{dataset_name}/{version_control}/location_error"
        evaluation_folder = f"{dataset_name}/{version_control}/evaluation_errors"

        with open(file, "r") as f:
            valid_errors = json.load(f)

        for paper, c in valid_errors.items():
            claim_file = f"{claim_folder}/{paper}_{model_insertion}.txt"
            inserted_error_file = f"{error_folder}/{paper}_{c}_{model_insertion}.txt"
            located_error_file = f"{location_folder}/{paper}_{c}_{model_insertion}.txt"
            identified_error_file = (
                f"{evaluation_folder}/{paper}_{c}_{model_identification}.txt"
            )
            claim = format_claims(claim_file)[int(c)]

            with open(inserted_error_file, "r") as f:
                inserted_error = f.read()

            with open(located_error_file, "r") as f:
                located_error = f.read()

            with open(identified_error_file, "r") as f:
                identified_error = f.read()

            row = {
                "paper": paper,
                "insertion_model": model_insertion,
                "identification_model": model_identification,
                "claim": claim,
                "claim_index": c,
                "inserted_error": inserted_error,
                "located_error": located_error,
                "identified_error": identified_error,
            }

            all_results.append(row)

    csv_path = f"{dataset_name}/all_results.csv"

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)

    print(f"Saved {len(all_results)} rows to {csv_path}")


if __name__ == "__main__":
    """CHANGE THE IDENTIFICATION MODELS"""
    model_family_identification = "openai"
    model_identification = "gpt-5-2025-08-07"

    """DEFAULT EVALUATION PARAMETERS"""
    levenshtein_threshold = 0.5
    top_k = 10
    insertion_families = {"gemini-2.5-pro": "gemini", "gpt-5-2025-08-07": "openai"}
    dataset_name = "data"
    model_names = {
        "gemini-2.5-pro": ["ALL_GEMINI", "gemini_all"],
        "gpt-5-2025-08-07": ["ALL_OPENAI", "openai_all"],
    }

    for model_insertion, name in model_names.items():
        version_control = name[0]
        file = f"{dataset_name}/{version_control}/{name[1]}.json"

        # make the necessary result folders
        os.makedirs(f"{dataset_name}/{version_control}", exist_ok=True)
        os.makedirs(f"{dataset_name}/{version_control}/altered_papers", exist_ok=True)
        error_folder = f"{dataset_name}/{version_control}/inserted_error"
        location_folder = f"{dataset_name}/{version_control}/location_error"
        evaluation_folder = f"{dataset_name}/{version_control}/evaluation_errors"

        with open(file, "r") as f:
            valid_errors = json.load(f)
        identification_dict = defaultdict(list)

        for paper, c in valid_errors.items():
            identification = identify_and_evaluate(
                model_insertion=model_insertion,
                model_family_insertion=insertion_families[model_insertion],
                model_family_identification=model_family_identification,
                model_identification=model_identification,
                error_folder=error_folder,
                location_folder=location_folder,
                external_identify_folder=evaluation_folder,
                version_control=version_control,
                paper=paper,
                ind=c,
                rerun_external_identification=False,
                rerun_evaluation=False,
                levenshtein_threshold=levenshtein_threshold,
                top_k=top_k,
                dataset_name=dataset_name,
            )
            identification_dict[paper].append({c: identification})
        with open(
            f"{dataset_name}/{version_control}/{model_insertion}_external_evals_{model_identification}.json",
            "w",
        ) as f:
            json.dump(identification_dict, f)
        print(compute_accuracy(identification_dict))
    print("External identification and evaluation completed")

    generate_csv_results(
        model_names,
        dataset_name,
        model_identification,
    )
