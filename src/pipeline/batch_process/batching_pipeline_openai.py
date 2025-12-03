import os
import json
import time
import traceback
from typing import Literal
from collections import defaultdict

import openai
from openai import OpenAI
from src.pipeline.batch_process.helper import (
    run_stage_with_executor,
    generate_claim_jsonl,
    generate_error_jsonl,
    generate_error_filter_jsonl,
    generate_error_location_jsonl,
    generate_self_identification_jsonl,
    get_non_self_identified_errors,
    get_valid_errors,
    create_altered_sources,
    get_papers,
    split_jsonl_file,
    compile_pdfs,
)
from src.pipeline.single_process.single_error_identification import (
    identify_and_evaluate,
)

os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()


def wait_for_openai_batch(
    batch_id: str, folder: str, poll_interval: int = 300, max_wait_hours: int = 24
) -> None:
    """
    Wait for OpenAI batch job to finish.
    """
    start = time.time()
    while True:
        status = openai_client.batches.retrieve(batch_id)
        state = status.status
        if state == "completed":
            get_batch_results_openai(batch_id, folder)
            break
        elif state in ("failed", "expired"):
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status {state}")
        elif (time.time() - start) > max_wait_hours * 3600:
            raise TimeoutError(f"OpenAI batch {batch_id} exceeded {max_wait_hours}h")
        else:
            print(
                f"[OpenAI] Batch {batch_id} still {state}... retry in {poll_interval}s"
            )
            time.sleep(poll_interval)


def submit_batch_openai(
    jsonl_file: str,
    endpoint: Literal[
        "/v1/responses", "/v1/chat/completions", "/v1/embeddings", "/v1/completions"
    ] = "/v1/chat/completions",
) -> str:
    """
    Submit a batch for openai batch processing,
    return batch id.
    """
    try:
        batch_file = openai_client.files.create(
            file=open(jsonl_file, "rb"), purpose="batch"
        )
    except Exception:
        traceback.print_exc()
        raise
    batch = openai_client.batches.create(
        input_file_id=batch_file.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    with open("data/temp_openai_batch_id.txt", "w") as f:
        f.write(batch.id)
    return batch.id


def get_batch_results_openai(batch_id: str, folder: str) -> None:
    """
    Retrieve the batch processing results from openai,
    and save it into individual files.
    """
    if not batch_id:
        with open("data/temp_openai_batch_id.txt", "r") as f:
            batch_id = f.read()
    status = openai_client.batches.retrieve(batch_id)
    state = status.status
    if state != "completed":
        print(f"Batch {batch_id} not ready: {state}")
        return
    output_file_id = status.output_file_id
    if not output_file_id:
        print("Error: no status.output_file_id")
        return
    file_content = openai_client.files.content(output_file_id).text
    with open(f"{folder}/openai_results.txt", "w") as f:
        f.write(file_content)
    with open(f"{folder}/openai_results.txt", "r") as f:
        for line in f:
            data = json.loads(line)
            key = data["custom_id"]
            model = data["response"]["body"]["model"]
            response_text = data["response"]["body"]["choices"][0]["message"]["content"]
            with open(f"{folder}/{key}_{model}.txt", "w") as f:
                f.write(response_text)


if __name__ == "__main__":
    """CHANGE INSERTION MODEL"""
    model_family_insertion = "openai"
    model_insertion = "gpt-5-2025-08-07"

    """CHANGE THIS FOR VERSION CONTROL"""
    version_control = "test_batch"

    # make the necessary result folders
    os.makedirs(f"data/{version_control}", exist_ok=True)
    os.makedirs(f"data/{version_control}/altered_papers", exist_ok=True)
    claim_folder = f"data/{version_control}/generated_claims"
    error_folder = f"data/{version_control}/inserted_error"
    filter_folder = f"data/{version_control}/filtered_error"
    location_folder = f"data/{version_control}/location_error"
    identify_folder = f"data/{version_control}/identified_errors"
    evaluation_folder = f"data/{version_control}/evaluation_errors"

    """CHANGE PAPER FORLDER AND EVALUATION PARAMETERS"""
    papers = get_papers("data/papers/")
    hallucination_threshold = 0.9
    levenshtein_threshold = 0.5
    top_k = 10

    ### ---------- Claim Extraction ---------- ###
    jsonl_file_openai = f"{claim_folder}/{model_insertion}_claim_requests.jsonl"

    generate_claim_jsonl(
        model_family=model_family_insertion,
        jsonl_filename=jsonl_file_openai,
        papers=papers,
        claim_folder=claim_folder,
        model=model_insertion,
    )

    if os.path.getsize(jsonl_file_openai) != 0:
        batch_id = submit_batch_openai(jsonl_file_openai)
        run_stage_with_executor(
            wait_for_openai_batch,
            claim_folder,
            batch_id,
            "Claim Extraction",
        )

    ### ---------- Error Generation ---------- ###
    jsonl_file_openai = f"{error_folder}/{model_insertion}_error_requests.jsonl"

    claims = generate_error_jsonl(
        jsonl_filename=jsonl_file_openai,
        papers=papers,
        claim_folder=claim_folder,
        error_folder=error_folder,
        model=model_insertion,
        model_family=model_family_insertion,
    )

    if os.path.getsize(jsonl_file_openai) != 0:
        split_files = split_jsonl_file(jsonl_file_openai, lines_per_file=2000)

        for part_file in split_files:
            print(part_file)
            batch_id = submit_batch_openai(part_file)
            run_stage_with_executor(
                wait_for_openai_batch,
                error_folder,
                batch_id,
                f"Error Generation (batch part: {os.path.basename(part_file)})",
            )

    ### ---------- Error Filter ---------- ###
    jsonl_file_openai = f"{filter_folder}/{model_insertion}_filter_requests.jsonl"

    generate_error_filter_jsonl(
        jsonl_filename=jsonl_file_openai,
        filter_folder=filter_folder,
        error_folder=error_folder,
        model=model_insertion,
        model_family=model_family_insertion,
    )

    if os.path.getsize(jsonl_file_openai) != 0:
        split_files = split_jsonl_file(jsonl_file_openai, lines_per_file=2000)
        for part_file in split_files:
            print(part_file)
            batch_id = submit_batch_openai(part_file)
            run_stage_with_executor(
                wait_for_openai_batch,
                filter_folder,
                batch_id,
                f"Error Filter (batch part: {os.path.basename(part_file)})",
            )

        valid_errors = get_valid_errors(error_folder, filter_folder, model_insertion)

    ### ---------- Error Location ---------- ###
    jsonl_file_openai = f"{location_folder}/{model_insertion}_location_requests.jsonl"

    generate_error_location_jsonl(
        jsonl_filename=jsonl_file_openai,
        error_folder=error_folder,
        filter_folder=filter_folder,
        location_folder=location_folder,
        model=model_insertion,
        model_family=model_family_insertion,
    )

    if os.path.getsize(jsonl_file_openai) != 0:
        batch_id = submit_batch_openai(jsonl_file_openai)

        run_stage_with_executor(
            wait_for_openai_batch,
            location_folder,
            batch_id,
            "Error Location",
        )

    ### ---------- Error Internal Identification ---------- ###
    create_altered_sources(
        error_folder=error_folder,
        filter_folder=filter_folder,
        model=model_insertion,
        hallucination_threshold=hallucination_threshold,
        version_control=version_control,
    )

    jsonl_file_openai = (
        f"{identify_folder}/{model_insertion}_self_identify_request.jsonl"
    )

    generate_self_identification_jsonl(
        jsonl_filename=jsonl_file_openai,
        error_folder=error_folder,
        location_folder=location_folder,
        identify_folder=identify_folder,
        filter_folder=filter_folder,
        model=model_insertion,
        model_family=model_family_insertion,
        version_control=version_control,
    )

    if os.path.getsize(jsonl_file_openai) != 0:
        split_files = split_jsonl_file(jsonl_file_openai, lines_per_file=2000)

        for part_file in split_files:
            print(part_file)
            batch_id = submit_batch_openai(part_file)
            run_stage_with_executor(
                wait_for_openai_batch,
                identify_folder,
                batch_id,
                f"Self Identification (batch part: {os.path.basename(part_file)})",
            )

    remaining_errors = get_non_self_identified_errors(
        model=model_insertion,
        identify_folder=identify_folder,
        error_folder=error_folder,
        location_folder=location_folder,
        filter_folder=filter_folder,
        threshold=levenshtein_threshold,
        top_k=top_k,
    )
    compile_pdfs(
        identify_folder=identify_folder,
        model=model_insertion,
        version_control=version_control,
    )

    print("🎉 Full insertion pipeline finished successfully!")

    ### ---------- Error External Identification ---------- ###
    """CHANGE THE EXTERNAL IDENTIFICATION MODELS"""
    model_family_identification = "deepseek"
    model_identification = "deepseek-reasoner"

    with open(f"{identify_folder}/{model_insertion}_compiled_pdfs.json", "r") as f:
        valid_errors = json.load(f)
    identification_dict = defaultdict(list)
    for paper, ind_list in valid_errors.items():
        for c in ind_list:
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
            )
            identification_dict[paper].append({c: identification})
    with open(
        f"data/{version_control}/{model_insertion}_external_evals.json", "w"
    ) as f:
        json.dump(identification_dict, f)
    print("External identification and evaluation completed")
