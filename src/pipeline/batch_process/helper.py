import os
import json
from typing import Callable
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.utils.formatting import (
    format_claims,
    format_generated_error,
    format_identified_error,
    format_localized_error,
)
from src.utils.prompts import (
    generate_claim_prompt,
    generate_error_generation_prompt,
    generate_filter_invalid_error_prompt,
    generate_filter_easy_error_prompt,
    generate_error_location_prompt,
    generate_internal_error_identification_prompt,
)
from src.utils.latex_to_pdf import (
    combine_latex_sources,
    compile_latex,
    check_pdf_for_unresolved_references,
    compress_pdf_ghostscript,
)
from src.utils.insertion_helpers import modify_source, create_altered_latex
from src.utils.evaluation_helpers import levenshtein_identify


def get_papers(folder: str) -> list[str]:
    """
    Get all papers in paper folder,
    return list of papers.
    """
    return [
        name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))
    ]


def run_stage_with_executor(
    func: Callable, folder: str, batch_id: str, stage_name: str
) -> None:
    """
    Runs OpenAI batch polling concurrently and waits for both.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(func, batch_id, folder),
        ]
        for future in as_completed(futures):
            try:
                # propagate exception if any
                future.result()
            except Exception as e:
                print(f"❌ {stage_name} job failed: {e}")
    print(f"✅ {stage_name} stage complete → results saved in {folder}")


def split_jsonl_file(
    input_path: str, lines_per_file: int = 50000, output_prefix: str | None = None
) -> list[str]:
    """
    Split jsonl files into smaller files because the apis can't accepts file that are too big,
    return all splitted filenames.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    if output_prefix is None:
        base, _ = os.path.splitext(input_path)
        output_prefix = base

    output_files = []
    file_index = 0
    line_count = 0

    output_path = f"{output_prefix}_part{file_index + 1}.jsonl"
    output_files.append(output_path)
    out_file = open(output_path, "w", encoding="utf-8")

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if line_count > 0 and line_count % lines_per_file == 0:
                out_file.close()
                file_index += 1
                output_path = f"{output_prefix}_part{file_index + 1}.jsonl"
                output_files.append(output_path)
                out_file = open(output_path, "w", encoding="utf-8")

            out_file.write(line)
            line_count += 1

    out_file.close()
    print(f"Split complete: {len(output_files)} files created.")
    return output_files


def format_input_message(
    model_family: str, id: str, full_prompt: str, model: str | None = None
) -> dict:
    """
    Format the input message to API calls depending on model,
    return the input message.
    """
    if model_family == "openai":
        message = {
            "custom_id": id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [{"role": "user", "content": full_prompt}],
            },
        }
    elif model_family == "gemini":
        message = {
            "key": id,
            "request": {"contents": [{"parts": [{"text": full_prompt}]}]},
        }
    else:
        message = {}
    return message


def generate_claim_jsonl(
    jsonl_filename: str,
    papers: list[str],
    claim_folder: str,
    model_family: str,
    model: str,
) -> None:
    """
    Make jsonl file for claim generation.
    """
    os.makedirs(claim_folder, exist_ok=True)
    prompt = generate_claim_prompt()

    with open(jsonl_filename, "w") as json_f:
        for paper in papers:
            if os.path.exists(f"{claim_folder}/{paper}_{model}.txt"):
                print("Already done this paper", paper)
                continue

            latex_file = combine_latex_sources(f"data/papers/{paper}")
            with open(latex_file, "r") as f:
                latex = f.read()

            full_prompt = prompt + "\n\nInput Latex Source:\n" + latex

            message = format_input_message(
                model_family=model_family,
                model=model,
                id=paper,
                full_prompt=full_prompt,
            )
            json_f.write(json.dumps(message) + "\n")


def generate_error_jsonl(
    jsonl_filename: str,
    papers: list[str],
    error_folder: str,
    claim_folder: str,
    model: str,
    model_family: str,
) -> dict[str, list[str]]:
    """
    Make error generation jsonl and also get a dictionary of all claims,
    return the dictionary.
    """
    os.makedirs(error_folder, exist_ok=True)
    claims = {}
    with open(jsonl_filename, "w") as json_f:
        for paper in papers:
            claim_filename = f"{claim_folder}/{paper}_{model}.txt"
            latex_file = f"data/papers/{paper}/combined.tex"
            with open(latex_file, "r") as f:
                latex = f.read()
            claim_list = format_claims(claim_filename)
            claims[paper] = claim_list
            for c, claim in enumerate(claim_list):
                error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
                if os.path.exists(error_filename):
                    print("Already done this paper", paper)
                    continue
                prompt = generate_error_generation_prompt(claim)

                full_prompt = prompt + "\n\nInput Latex Source:\n" + latex

                message = format_input_message(
                    model_family=model_family,
                    model=model,
                    id=f"{paper}_{c}",
                    full_prompt=full_prompt,
                )

                json_f.write(json.dumps(message) + "\n")
    with open(f"{error_folder}/{model}_all_claims.json", "w") as f:
        f.write(json.dumps(claims))
    return claims


def generate_error_filter_jsonl(
    jsonl_filename: str,
    error_folder: str,
    filter_folder: str,
    model: str,
    model_family: str,
) -> None:
    """
    Generate jsonl to filter through errors.
    """
    os.makedirs(filter_folder, exist_ok=True)
    with open(f"{error_folder}/{model}_all_claims.json", "r") as file:
        all_claims = json.load(file)

    with open(jsonl_filename, "w") as json_f:
        for paper, claim_list in all_claims.items():
            latex_file = f"data/papers/{paper}/combined.tex"
            with open(latex_file, "r") as f:
                latex = f.read()
            for c, claim in enumerate(claim_list):
                error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
                # skip claims where an error was not generated
                if not os.path.exists(error_filename):
                    continue
                modified_text, original_text, explanation, _ = format_generated_error(
                    error_filename
                )
                invalid_filter_filename = (
                    f"{filter_folder}/{paper}_{c}_invalid_{model}.txt"
                )
                easy_filter_filename = f"{filter_folder}/{paper}_{c}_easy_{model}.txt"
                if not os.path.exists(invalid_filter_filename):
                    prompt = generate_filter_invalid_error_prompt(
                        original_text=original_text,
                        modified_text=modified_text,
                        broken_claims=[claim],
                        explanation_list=explanation,
                    )

                    full_prompt = prompt + "\n\nInput Latex Source:\n" + latex

                    message = format_input_message(
                        model_family=model_family,
                        model=model,
                        id=f"{paper}_{c}_invalid",
                        full_prompt=full_prompt,
                    )
                    json_f.write(json.dumps(message) + "\n")

                if not os.path.exists(easy_filter_filename):
                    error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
                    prompt = generate_filter_easy_error_prompt(
                        original_text=original_text,
                        modified_text=modified_text,
                        broken_claims=[claim],
                        explanation_list=explanation,
                    )

                    full_prompt = prompt + "\n\nInput Latex Source:\n" + latex
                    message = format_input_message(
                        model_family=model_family,
                        model=model,
                        id=f"{paper}_{c}_easy",
                        full_prompt=full_prompt,
                    )
                    json_f.write(json.dumps(message) + "\n")
                print(f"Already done claim {c} of {paper}")


def get_valid_errors(
    error_folder: str, filter_folder: str, model: str
) -> dict[str, list[int | str]]:
    """
    Get all valid errors after filtering,
    return a dictionary of all valid errors.
    """
    with open(f"{error_folder}/{model}_all_claims.json", "r") as file:
        all_claims = json.load(file)
    valid_errors = defaultdict(list)
    for paper, claim_list in all_claims.items():
        for c, _ in enumerate(claim_list):
            invalid_filter_filename = f"{filter_folder}/{paper}_{c}_invalid_{model}.txt"
            easy_filter_filename = f"{filter_folder}/{paper}_{c}_easy_{model}.txt"
            if not check_need_change(invalid_filter_filename) and not check_need_change(
                easy_filter_filename
            ):
                valid_errors[paper].append(c)
    with open(f"{filter_folder}/{model}_valid_errors_after_filtering.json", "w") as f:
        json.dump(valid_errors, f)
    return valid_errors


def check_need_change(filename: str) -> bool:
    """
    Check if the error is too easy or invlaid given a result file.
    """
    if os.path.exists(filename):
        with open(filename, "r") as f:
            content = f.read()
        if "No changes required" in content:
            return False
        return True
    else:
        return True


def generate_error_location_jsonl(
    jsonl_filename: str,
    model_family: str,
    error_folder: str,
    filter_folder: str,
    location_folder: str,
    model: str,
) -> None:
    """
    Create jsonl file for localizing the errors.
    """
    os.makedirs(location_folder, exist_ok=True)
    with open(
        f"{filter_folder}/{model}_valid_errors_after_filtering.json", "r"
    ) as file:
        valid_errors = json.load(file)

    with open(f"{error_folder}/{model}_all_claims.json", "r") as file:
        all_claims = json.load(file)

    with open(jsonl_filename, "w") as json_f:
        for paper, claim_list in valid_errors.items():
            latex_file = f"data/papers/{paper}/combined.tex"
            with open(latex_file, "r") as f:
                latex_source = f.read()
            for c in claim_list:
                error_location_file = f"{location_folder}/{paper}_{c}_{model}.txt"
                if os.path.exists(error_location_file):
                    print("Already done this paper", paper)
                    continue
                error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
                modified_text, original_text, explanation, _ = format_generated_error(
                    error_filename
                )
                claim = all_claims[paper][c]
                prompt = generate_error_location_prompt(
                    original_text=original_text,
                    modified_text=modified_text,
                    broken_claims=[claim],
                    explanation_list=explanation,
                )
                full_prompt = prompt + "\n\nInput Latex Source:\n" + latex_source
                message = format_input_message(
                    model_family=model_family,
                    model=model,
                    id=f"{paper}_{c}",
                    full_prompt=full_prompt,
                )
                json_f.write(json.dumps(message) + "\n")


def create_altered_sources(
    error_folder: str,
    filter_folder: str,
    model: str,
    hallucination_threshold: float,
    version_control: str,
) -> None:
    """
    Create the altered sources of all valid paper-error pairs.
    """
    with open(f"{filter_folder}/{model}_valid_errors_after_filtering.json", "r") as f:
        valid_errors = json.load(f)

    for paper, claim_list in valid_errors.items():
        latex_file = f"data/papers/{paper}/combined.tex"
        with open(latex_file, "r") as f:
            latex_source = f.read()
        for c in claim_list:
            error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
            altered_source = modify_source(
                error_filename=error_filename,
                latex_source=latex_source,
                threshold=hallucination_threshold,
            )
            if altered_source is None:
                print("Failed to insert error into the latex source.")
                continue

            altered_tex_filename = (
                f"data/{version_control}/altered_papers/{paper}/altered_{c}.tex"
            )
            if not os.path.exists(altered_tex_filename):
                create_altered_latex(
                    paper=paper,
                    new_latex=altered_source,
                    version=version_control,
                    new_main_file=f"altered_{c}",
                )


def generate_self_identification_jsonl(
    jsonl_filename: str,
    error_folder: str,
    location_folder: str,
    identify_folder: str,
    filter_folder: str,
    model: str,
    version_control: str,
    model_family: str,
) -> None:
    """
    Generate jsonl file for self-identification.
    """
    os.makedirs(identify_folder, exist_ok=True)
    with open(
        f"{filter_folder}/{model}_valid_errors_after_filtering.json", "r"
    ) as file:
        valid_errors = json.load(file)

    with open(jsonl_filename, "w") as json_f:
        for paper, claim_list in valid_errors.items():
            for c in claim_list:
                identify_filename = f"{identify_folder}/{paper}_{c}_{model}.txt"
                if os.path.exists(identify_filename):
                    print("Already done this paper", paper)
                    continue
                error_location_filename = f"{location_folder}/{paper}_{c}_{model}.txt"
                if not os.path.exists(error_location_filename):
                    print("Error was not localized.")
                    continue

                error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
                modified_text, original_text, _, _ = format_generated_error(
                    error_filename
                )
                localized_errors = format_localized_error(error_location_filename)
                word_limit = max(
                    [len(i.split()) for i in modified_text]
                    + [len(i.split()) for i in localized_errors]
                    + [len(i.split()) for i in original_text]
                )

                prompt = generate_internal_error_identification_prompt(
                    num_chunks=10, word_limit=word_limit
                )
                altered_latex_filename = (
                    f"data/{version_control}/altered_papers/{paper}/altered_{c}.tex"
                )
                with open(altered_latex_filename, "r") as f:
                    latex = f.read()

                full_prompt = prompt + "\n\nInput Latex Source:\n" + latex

                message = format_input_message(
                    model_family=model_family,
                    model=model,
                    id=f"{paper}_{c}",
                    full_prompt=full_prompt,
                )
                json_f.write(json.dumps(message) + "\n")


def get_non_self_identified_errors(
    identify_folder: str,
    error_folder: str,
    location_folder: str,
    filter_folder: str,
    model: str,
    threshold: float,
    top_k: int,
) -> dict[str, list[int | str]]:
    """
    Get all valid errors after self-identification,
    return a dictionary of all non-identified errors.
    """
    with open(
        f"{filter_folder}/{model}_valid_errors_after_filtering.json", "r"
    ) as file:
        valid_errors = json.load(file)
    print(valid_errors)

    remaining_errors = defaultdict(list)
    for paper, claim_list in valid_errors.items():
        for c in claim_list:
            identify_error_filename = f"{identify_folder}/{paper}_{c}_{model}.txt"

            if not os.path.exists(identify_error_filename):
                print("Error has not been self-identified.")
                continue

            error_filename = f"{error_folder}/{paper}_{c}_{model}.txt"
            error_location_filename = f"{location_folder}/{paper}_{c}_{model}.txt"

            if not os.path.exists(error_filename) or not os.path.exists(
                error_location_filename
            ):
                print("No error file or location file.")
                continue
            modified_text, _, _, _ = format_generated_error(error_filename)
            true_error_list = modified_text + format_localized_error(
                error_location_filename
            )
            pred_error_list = format_identified_error(identify_error_filename)
            regeneration = levenshtein_identify(
                folder=identify_folder,
                paper=paper,
                ind=c,
                model=model,
                true_error_list=true_error_list,
                pred_error_list=pred_error_list,
                threshold=threshold,
                top_k=top_k,
            )
            if not regeneration:
                remaining_errors[paper].append(c)
    with open(
        f"{identify_folder}/{model}_remaining_error_after_self_identification.json", "w"
    ) as f:
        json.dump(remaining_errors, f)
    return remaining_errors


def compile_pdfs(
    identify_folder: str, model: str, version_control: str
) -> dict[str, list[str | int]]:
    """Compile the pdf for the valid errors after self identification,
    return the dictionary of valid errors."""
    with open(
        f"{identify_folder}/{model}_remaining_error_after_self_identification.json", "r"
    ) as f:
        valid_errors = json.load(f)
    valid_pdfs = defaultdict(list)
    for paper, claim_inds in valid_errors.items():
        for c in claim_inds:
            altered_tex_filename = (
                f"data/{version_control}/altered_papers/{paper}/altered_{c}.tex"
            )
            altered_pdf_filename = (
                f"data/{version_control}/altered_papers/{paper}/altered_{c}.pdf"
            )
            compressed_pdf_filename = (
                f"data/{version_control}/altered_papers/{paper}/altered_{c}_small.pdf"
            )

            if not os.path.exists(altered_pdf_filename):
                _ = compile_latex(
                    paper=paper,
                    des=f"/data/{version_control}/altered_papers/{paper}",
                    main_tex=altered_tex_filename,
                )

            valid_pdf = check_pdf_for_unresolved_references(altered_pdf_filename)
            if not valid_pdf:
                print(f"PDF for claim {c} of {paper} cannot be successfully compiled")
                continue

            if not os.path.exists(compressed_pdf_filename):
                compress_pdf_ghostscript(
                    altered_pdf_filename,
                    compressed_pdf_filename,
                )
            if os.path.exists(compressed_pdf_filename):
                valid_pdfs[paper].append(c)
    with open(f"{identify_folder}/{model}_compiled_pdfs.json", "w") as f:
        json.dump(valid_pdfs, f)
    return valid_pdfs
