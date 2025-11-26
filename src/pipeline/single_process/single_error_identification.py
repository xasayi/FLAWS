import os
from typing import Callable

from src.utils.llm_calls import call_api, completion_response
from src.utils.formatting import (
    format_generated_error,
    format_localized_error,
    format_identified_error,
)

from src.utils.evaluation_helpers import (
    levenshtein_identify,
    parse_llm_as_a_judge,
    parse_levenshtein,
)

from src.utils.prompts import (
    generate_error_identification_prompt,
    generate_error_evaluation_prompt,
)


def identify_error(
    paper: str,
    prompt: Callable,
    model_family: str,
    model: str,
    identify_folder: str,
    pdf: str | None,
    ind: int | str,
    num_chunks: int = 10,
    word_limit: int = 120,
) -> str:
    """
    External identification of inserted error,
    return filename of where identified error is saved.
    """
    prompt_find_err = prompt(num_chunks=num_chunks, word_limit=word_limit)
    completion, _ = call_api(
        prompt=prompt_find_err,
        model=model,
        folder=identify_folder,
        paper=paper,
        model_family=model_family,
        latex_file=None,
        pdf_path=pdf,
    )
    with open(f"{identify_folder}/{paper}_{ind}_{model}.txt", "w") as f:
        f.write(completion)
        filename = f.name
    print(f"Saved identified error at {filename}")
    return filename


def evaluate_error(
    paper: str,
    model_family_identification: str,
    model_identification: str,
    identified_error_folder: str,
    identify_error_filename: str,
    prompt: Callable,
    modified_text: list[str],
    error_location_filename: str,
    threshold: float,
    ind: str | int,
    top_k: int,
) -> bool:
    """
    Evaluate external error identification using Levenshtein distance and LLM as a judge,
    return whether the error was identified by one of them or not.
    """
    if identified_error_folder is not None:
        os.makedirs(identified_error_folder, exist_ok=True)
    true_error_list = modified_text + format_localized_error(error_location_filename)
    pred_error_list = format_identified_error(identify_error_filename)

    levenshtein_regeneration = levenshtein_identify(
        folder=identified_error_folder,
        paper=paper,
        ind=ind,
        model=model_identification,
        true_error_list=true_error_list,
        pred_error_list=pred_error_list,
        threshold=threshold,
        top_k=top_k,
    )
    input_prompt = prompt(
        true_error_text=true_error_list, pred_error_text=pred_error_list
    )

    llm_response = completion_response(
        prompt=input_prompt,
        model=model_identification,
        model_family=model_family_identification,
        pdf_path=None,
    )

    with open(
        f"{identified_error_folder}/{paper}_{ind}_{model_identification}_comparison.txt",
        "w",
    ) as f:
        f.write(llm_response)
        filename = f.name
    llm_regeneration = parse_llm_as_a_judge(llm_response, top_k)
    print(f"Saved error evaluation at {filename}")
    return levenshtein_regeneration or llm_regeneration


def identify_and_evaluate(
    model_insertion: str,
    model_identification: str,
    model_family_identification: str,
    error_folder: str,
    external_identify_folder: str,
    version_control: str,
    location_folder: str,
    paper: str,
    ind: str | int,
    rerun_external_identification: bool,
    rerun_evaluation: bool,
    levenshtein_threshold: float,
    top_k: int,
    dataset_name: str = "data",
) -> bool:
    """
    Identify and evaluate the error for a single paper, ind pair,
    return whether identification was successful or not.
    """
    # Define filenames
    error_filename = f"{error_folder}/{paper}_{ind}_{model_insertion}.txt"
    error_location_filename = f"{location_folder}/{paper}_{ind}_{model_insertion}.txt"
    external_identify_filename = (
        f"{external_identify_folder}/{paper}_{ind}_{model_identification}.txt"
    )
    external_identify_score_filename = (
        f"{external_identify_folder}/{paper}_{ind}_{model_identification}_score.txt"
    )
    external_identify_llm_filename = f"{external_identify_folder}/{paper}_{ind}_{model_identification}_comparison.txt"
    compressed_pdf_filename = f"{dataset_name}/{version_control}/altered_papers/{paper}/altered_{ind}_small.pdf"
    # Start automated error evaluation framework,
    # at each step check whether is had been done or not.
    if not os.path.exists(error_filename):
        print(
            f"error has not been generated for claim {ind} of {paper}", error_filename
        )

    if not os.path.exists(error_location_filename):
        print(f"error has not been localized for claim {ind} of {paper}")
        exit()
    if not os.path.exists(compressed_pdf_filename):
        print("the compressed pdf does not exist")
        compressed_pdf_filename = (
            f"{dataset_name}/{version_control}/altered_papers/{paper}/altered_{ind}.pdf"
        )

    print(external_identify_filename, rerun_external_identification)
    if not os.path.exists(external_identify_filename) or rerun_external_identification:
        modified_text, original_text, _, _ = format_generated_error(error_filename)
        localized_errors = format_localized_error(error_location_filename)
        word_limit = max(
            [len(i.split()) for i in modified_text]
            + [len(i.split()) for i in localized_errors]
            + [len(i.split()) for i in original_text]
        )
        _ = identify_error(
            paper=paper,
            prompt=generate_error_identification_prompt,
            model_family=model_family_identification,
            model=model_identification,
            identify_folder=external_identify_folder,
            pdf=compressed_pdf_filename,
            ind=ind,
            num_chunks=top_k,
            word_limit=word_limit,
        )
    print("1-------Error is Externally Identified-------1")

    successful_identification = False
    if (
        not os.path.exists(external_identify_score_filename)
        or not os.path.exists(external_identify_llm_filename)
        or rerun_evaluation
    ):
        modified_text, _, _, _ = format_generated_error(error_filename)
        successful_identification = evaluate_error(
            paper=paper,
            model_family_identification=model_family_identification,
            model_identification=model_identification,
            identified_error_folder=external_identify_folder,
            identify_error_filename=external_identify_filename,
            prompt=generate_error_evaluation_prompt,
            modified_text=modified_text,
            error_location_filename=error_location_filename,
            threshold=levenshtein_threshold,
            ind=ind,
            top_k=top_k,
        )
    else:
        try:
            with open(external_identify_score_filename, "r") as f:
                leven_content = f.read()
            with open(external_identify_llm_filename, "r") as f:
                llm_content = f.read()
            successful_identification = bool(
                parse_levenshtein(leven_content, top_k, threshold=levenshtein_threshold)
            ) or parse_llm_as_a_judge(llm_content, top_k)
        except Exception:
            successful_identification = False

    print("2----External Identification is Evaluated----2")
    return successful_identification


if __name__ == "__main__":
    """CHANGE THIS FOR SPECIFIC IDENTIFICATION MODEL"""
    model_insertion = "gemini-2.5-pro"
    model_family_identification = "deepseek"
    model_identification = "deepseek-reasoner"

    """CHANGE THIS FOR VERSION CONTROL"""
    version_control = "cleanup"

    error_folder = f"data/{version_control}/inserted_error"
    location_folder = f"data/{version_control}/location_error"
    external_identify_folder = f"data/{version_control}/external_identify"

    """CHANGE THESE DEPENDIN ON WHAT YOU NEED TO RERUN"""
    rerun_external_identification = False
    rerun_evaluation = False

    """CHANGE THIS TO DETERMINE WHEN AN ERROR SHOULD BE DISCARDED"""
    # if value above this, we consider internal identification successful
    levenshtein_threshold = 0.5
    # number of top error text excerpts to use
    top_k = 10

    """CHANGE THESE FOR SPECIFIC PAPER AND CLAIM"""
    papers = ["Matchmaker"]
    ind = [1]

    for p, c in zip(papers, ind):
        # Define filenames
        identification = identify_and_evaluate(
            model_insertion=model_insertion,
            model_family_identification=model_family_identification,
            model_identification=model_identification,
            error_folder=error_folder,
            location_folder=location_folder,
            external_identify_folder=external_identify_folder,
            version_control=version_control,
            paper=p,
            ind=c,
            rerun_external_identification=rerun_external_identification,
            rerun_evaluation=rerun_evaluation,
            levenshtein_threshold=levenshtein_threshold,
            top_k=top_k,
        )
        print(f"Is the error successfully identified? {identification}")
