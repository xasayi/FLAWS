import os
from typing import Callable

from src.utils.llm_calls import call_api
from src.utils.formatting import (
    format_claims,
    format_generated_error,
    format_localized_error,
    format_identified_error,
)
from src.utils.prompts import (
    generate_claim_prompt,
    generate_error_generation_prompt,
    generate_filter_invalid_error_prompt,
    generate_filter_easy_error_prompt,
    generate_error_location_prompt,
    generate_internal_error_identification_prompt,
)
from src.utils.insertion_helpers import (
    create_altered_latex,
    modify_source,
)
from src.utils.evaluation_helpers import (
    levenshtein_identify,
    parse_levenshtein,
)
from src.utils.latex_to_pdf import (
    compile_latex,
    combine_latex_sources,
    find_main_tex_file_to_combine,
    check_pdf_for_unresolved_references,
    compress_pdf_ghostscript,
)


def extract_claims(
    paper: str,
    prompt: str,
    model_family: str,
    model: str,
    claim_folder: str,
    latex_file: str,
) -> int:
    """
    Extract claims from a paper using an LLM,
    return number of claims extracted
    """
    completion, _ = call_api(
        prompt=prompt,
        model=model,
        folder=claim_folder,
        paper=paper,
        model_family=model_family,
        latex_file=latex_file,
        pdf_path=None,
    )

    with open(f"{claim_folder}/{paper}_{model}.txt", "w") as f:
        f.write(completion)
        filename = f.name
    print(f"Saved generated completion at {filename}")

    if completion:
        claim_list = format_claims(filename)
        return len(claim_list)
    return 0


def generate_error(
    paper: str,
    prompt: Callable,
    model_family: str,
    model: str,
    claim_file: str,
    error_folder: str,
    latex_file: str,
    ind: int,
) -> str:
    """
    Generate an error for a specific claim in a paper,
    return filename of where the error is saved.
    """
    claim_list = format_claims(claim_file)

    input_prompt = prompt(claim_list[ind])

    completion, _ = call_api(
        prompt=input_prompt,
        model=model,
        folder=error_folder,
        paper=paper,
        model_family=model_family,
        latex_file=latex_file,
        pdf_path=None,
    )

    with open(f"{error_folder}/{paper}_{ind}_{model}.txt", "w") as f:
        f.write(completion)
        f.write("\n\n:claim:\n")
        f.write(claim_list[ind])
        filename = f.name
    print(f"Saved generated completion at {filename}")

    return filename


def filter_error(
    paper: str,
    prompt: Callable,
    model_family: str,
    model: str,
    error_folder: str,
    error_filename: str,
    latex_file: str,
    ind: str | int,
) -> bool:
    """
    Filter our easy or explicitly stated errors using LLM,
    return whether the error was too easy or explicitly stated.
    """
    modified_text, original_text, broken_claims, explanation_list = (
        format_generated_error(error_filename)
    )
    input_prompt = prompt(original_text, modified_text, broken_claims, explanation_list)

    completion, _ = call_api(
        prompt=input_prompt,
        paper=paper,
        model=model,
        model_family=model_family,
        folder=error_folder,
        latex_file=latex_file,
        pdf_path=None,
    )

    with open(f"{error_folder}/{paper}_{ind}_{model}_filter.txt", "w") as f:
        f.write(f"{completion}\n")
        filename = f.name
    print(f"Saved filtered error at {filename}")

    need_better_error = False
    if "No changes required" not in completion:
        need_better_error = True

    return need_better_error


def insert_error(
    paper: str,
    ind: str | int,
    version_control: str,
    error_filename: str,
    threshold: float,
    latex_file=None,
) -> str | None:
    """
    Insert the error into a the original LaTeX source,
    return the source or None if not successfully inserted.
    """
    altered_latex_filename = None
    if latex_file is not None:
        with open(latex_file, "r") as f:
            latex_source = f.read()
    else:
        latex_file = find_main_tex_file_to_combine(
            f"data/papers/{version_control}/{paper}"
        )
        with open(f"data/papers/{version_control}/{paper}/{latex_file}", "r") as f:
            latex_source = f.read()

    altered_latex_source = modify_source(
        error_filename=error_filename, latex_source=latex_source, threshold=threshold
    )
    if altered_latex_source:
        altered_latex_filename = create_altered_latex(
            paper=paper,
            new_latex=altered_latex_source,
            version=version_control,
            new_main_file=f"altered_{ind}",
        )
    return altered_latex_filename


def localize_error(
    paper: str,
    prompt: Callable,
    model_family: str,
    model: str,
    error_folder: str,
    error_filename: str,
    latex_file: str,
    ind: int,
) -> str:
    """
    Localize all error text excerpts,
    return the filename that contains the error text excerpts,
    """
    modified_text, original_text, broken_claims, explanation_list = (
        format_generated_error(error_filename)
    )
    input_prompt = prompt(original_text, modified_text, broken_claims, explanation_list)

    completion, _ = call_api(
        prompt=input_prompt,
        model=model,
        folder=error_folder,
        paper=paper,
        model_family=model_family,
        latex_file=latex_file,
        pdf_path=None,
    )

    with open(f"{error_folder}/{paper}_{ind}_{model}.txt", "w") as f:
        f.write(completion)
        filename = f.name
    print(f"Saved error location at {filename}")
    return filename


def internal_identify_error(
    paper: str,
    prompt: Callable,
    model_family: str,
    model: str,
    identify_folder: str,
    latex_file: str,
    ind: int,
    num_chunks: int = 10,
    word_limit: int = 120,
) -> str:
    """
    Internally identify the error,
    return the filename that contains the internal identificaiton results.
    """
    prompt_find_err = prompt(num_chunks=num_chunks, word_limit=word_limit)

    completion, _ = call_api(
        prompt=prompt_find_err,
        model=model,
        folder=identify_folder,
        paper=paper,
        model_family=model_family,
        latex_file=latex_file,
        pdf_path=None,
    )
    with open(f"{identify_folder}/{paper}_{ind}_{model}.txt", "w") as f:
        f.write(completion)
        filename = f.name
    print(f"Saved internal identified error at {filename}")
    return filename


def internal_evaluate_error(
    paper: str,
    error_location_filename: str,
    internal_identify_error_filename: str,
    identified_error_folder: str,
    modified_text: list[str],
    model: str,
    ind: int,
    threshold: float,
    generate_k: int,
) -> bool:
    """
    Evaluate whether the internal identification was successful,
    return whether a new error needs to be generated.
    """
    if identified_error_folder is not None:
        os.makedirs(identified_error_folder, exist_ok=True)
    true_error_list = modified_text + format_localized_error(error_location_filename)
    pred_error_list = format_identified_error(internal_identify_error_filename)

    regeneration = levenshtein_identify(
        folder=identified_error_folder,
        paper=paper,
        ind=ind,
        model=model,
        true_error_list=true_error_list,
        pred_error_list=pred_error_list,
        threshold=threshold,
        top_k=generate_k,
    )
    return regeneration


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    """CHANGE THIS FOR SPECIFIC INSERTION MODEL"""
    model_family_insertion = "gemini"
    model_insertion = "gemini-2.5-pro"

    """CHANGE THIS FOR VERSION CONTROL"""
    version_control = "test"

    os.makedirs(f"data/{version_control}", exist_ok=True)
    os.makedirs(f"data/{version_control}/altered_papers", exist_ok=True)
    claim_folder = f"data/{version_control}/generated_claims"
    error_folder = f"data/{version_control}/inserted_error"
    filter_folder = f"data/{version_control}/filtered_error"
    location_folder = f"data/{version_control}/location_error"
    identify_folder = f"data/{version_control}/identified_errors"

    """CHANGE THESE FOR SPECIFIC PAPER AND CLAIM"""
    paper = "Matchmaker"
    ind = 1

    """CHANGE THESE DEPENDIN ON WHAT YOU NEED TO RERUN"""
    reextract_claims = False
    regenerate_error = False
    refilter_error = False
    reinsert_error = False
    relocalize_error = False
    reidentify_error = False
    reeval_error = False
    recompile_pdf = False

    """CHANGE THIS TO DETERMINE WHEN AN ERROR SHOULD BE DISCARDED"""
    # if value above this, we regard the generation hallucinated
    hallucination_threshold = 0.9
    # if value above this, we consider internal identification successful
    levenshtein_threshold = 0.5
    # number of top error text excerpts to use
    generate_k = 10

    # Define filenames
    claim_filename = f"{claim_folder}/{paper}_{model_insertion}.txt"
    error_filename = f"{error_folder}/{paper}_{ind}_{model_insertion}.txt"
    error_filter_filename = f"{error_folder}/{paper}_{ind}_{model_insertion}_filter.txt"
    error_location_filename = f"{location_folder}/{paper}_{ind}_{model_insertion}.txt"
    internal_id_score_filename = (
        f"{identify_folder}/{paper}_{ind}_{model_insertion}_score.txt"
    )
    internal_id_filename = f"{identify_folder}/{paper}_{ind}_{model_insertion}.txt"
    altered_pdf_filename = (
        f"data/{version_control}/altered_papers/{paper}/altered_{ind}.pdf"
    )
    compressed_pdf_filename = (
        f"data/{version_control}/altered_papers/{paper}/altered_{ind}_small.pdf"
    )
    altered_tex_filename = (
        f"data/{version_control}/altered_papers/{paper}/altered_{ind}.tex"
    )
    latex_filename = f"data/papers/{paper}/combined.tex"

    # Start Automated Error Insertion pipeline,
    # at each step, we check whether the previous step has already been done
    # can also comment out specific parts
    if not os.path.exists(claim_filename) or reextract_claims:
        if not os.path.exists(latex_filename):
            latex_name = combine_latex_sources(f"data/papers/{paper}")
            print(latex_name, latex_filename)
            assert latex_name == latex_filename
        len_claims = 0
        while len_claims == 0:
            len_claims = extract_claims(
                paper=paper,
                prompt=generate_claim_prompt(),
                model_family=model_family_insertion,
                model=model_insertion,
                claim_folder=claim_folder,
                latex_file=latex_filename,
            )
            print(f"Generated {len_claims} claims")
    print("1---------Claims are Extracted---------1")

    if not os.path.exists(error_filename) or regenerate_error:
        filename = generate_error(
            paper=paper,
            prompt=generate_error_generation_prompt,
            model_family=model_family_insertion,
            model=model_insertion,
            claim_file=claim_filename,
            error_folder=error_folder,
            latex_file=latex_filename,
            ind=ind,
        )
        assert filename == error_filename
    print("2----------Error is Generated----------2")

    if not os.path.exists(error_filter_filename) or refilter_error:
        need_better_error = filter_error(
            paper=paper,
            prompt=generate_filter_invalid_error_prompt,
            model_family=model_family_insertion,
            model=model_insertion,
            error_folder=error_folder,
            error_filename=error_filename,
            latex_file=latex_filename,
            ind=ind,
        )
        if not need_better_error:
            need_better_error = filter_error(
                paper=paper,
                prompt=generate_filter_easy_error_prompt,
                model_family=model_family_insertion,
                model=model_insertion,
                error_folder=error_folder,
                error_filename=error_filename,
                latex_file=latex_filename,
                ind=ind,
            )
        if need_better_error:
            print("generated error is too easy, need to regenerate")
            exit()
    print("3--------Filtering is Completed--------3")

    if not os.path.exists(altered_tex_filename) or reinsert_error:
        altered_tex_filename = insert_error(
            paper=paper,
            error_filename=error_filename,
            version_control=version_control,
            latex_file=latex_filename,
            ind=ind,
            threshold=hallucination_threshold,
        )
        if altered_tex_filename is None:
            print("failed to insert Error due to original text mismatch")
            exit()
    print("4-----------Error is Inserted----------4")

    if not os.path.exists(error_location_filename) or relocalize_error:
        error_location_filename = localize_error(
            paper=paper,
            prompt=generate_error_location_prompt,
            model_family=model_family_insertion,
            model=model_insertion,
            error_folder=location_folder,
            error_filename=error_filename,
            latex_file=altered_tex_filename,
            ind=ind,
        )
    print("5----------Error is Localized----------5")
    if not os.path.exists(internal_id_filename) or reidentify_error:
        modified_text, original_text, explanation, broken_claim = (
            format_generated_error(error_filename)
        )
        localized_errors = format_localized_error(error_location_filename)
        word_limit = max(
            [len(i.split()) for i in modified_text]
            + [len(i.split()) for i in localized_errors]
            + [len(i.split()) for i in original_text]
        )
        internal_identify_error_filename = internal_identify_error(
            paper=paper,
            prompt=generate_internal_error_identification_prompt,
            model_family=model_family_insertion,
            model=model_insertion,
            identify_folder=identify_folder,
            latex_file=altered_tex_filename,
            num_chunks=generate_k,
            word_limit=word_limit,
            ind=ind,
        )
    print("6----Error is Internally Identified----6")

    regenerate_error = False
    if not os.path.exists(internal_id_score_filename) or reeval_error:
        modified_text, original_text, explanation, broken_claim = (
            format_generated_error(error_filename)
        )
        regenerate_error = internal_evaluate_error(
            paper=paper,
            error_location_filename=error_location_filename,
            internal_identify_error_filename=internal_id_filename,
            identified_error_folder=identify_folder,
            modified_text=modified_text,
            model=model_insertion,
            ind=ind,
            threshold=levenshtein_threshold,
            generate_k=generate_k,
        )
    else:
        with open(internal_id_score_filename, "r") as f:
            content = f.read()
        regenerate_error = bool(
            parse_levenshtein(content, generate_k, threshold=levenshtein_threshold)
        )

    if regenerate_error:
        print("Error has been self-identified, is it too easy.")
        exit()

    if not os.path.exists(altered_pdf_filename) or recompile_pdf:
        _ = compile_latex(
            paper=paper,
            des=f"/data/{version_control}/altered_papers/{paper}",
            main_tex=altered_tex_filename,
        )

    valid_pdf = check_pdf_for_unresolved_references(altered_pdf_filename)
    if not valid_pdf:
        print("PDF cannot be successfully compiled")

    if not os.path.exists(compressed_pdf_filename):
        compress_pdf_ghostscript(
            altered_pdf_filename,
            compressed_pdf_filename,
        )

    print("7------------PDF is compiled-----------7")
