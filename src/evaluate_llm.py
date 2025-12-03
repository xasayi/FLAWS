import os
import json
from collections import defaultdict
from src.pipeline.single_process.single_error_identification import (
    identify_and_evaluate,
)


def compute_accuracy(identification_dict: dict[str, list]) -> float:
    total = 0
    correct = 0
    for _, entries in identification_dict.items():
        for identify in entries:
            correct += list(identify.values())[0]
            total += 1

    return correct / total if total > 0 else 0


if __name__ == "__main__":
    """CHANGE THE IDENTIFICATION MODELS"""
    model_family_identification = "gpt"
    model_identification = "gpt-5-2025-08-07"

    """DEFAULT EVALUATION PARAMETERS"""
    levenshtein_threshold = 0.5
    top_k = 10

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
        claim_folder = f"{dataset_name}/{version_control}/generated_claims"
        error_folder = f"{dataset_name}/{version_control}/inserted_error"
        filter_folder = f"{dataset_name}/{version_control}/filtered_error"
        location_folder = f"{dataset_name}/{version_control}/location_error"
        identify_folder = f"{dataset_name}/{version_control}/identified_errors"
        evaluation_folder = f"{dataset_name}/{version_control}/evaluation_errors"

        with open(file, "r") as f:
            valid_errors = json.load(f)
        identification_dict = defaultdict(list)

        for paper, c in valid_errors.items():
            identification = identify_and_evaluate(
                model_insertion=model_insertion,
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
            f"{dataset_name}/{version_control}/{model_insertion}_external_evals.json",
            "w",
        ) as f:
            json.dump(identification_dict, f)
        print(compute_accuracy(identification_dict))
    print("External identification and evaluation completed")
