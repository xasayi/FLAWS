import os
import json
import time
import traceback
from collections import defaultdict

from google import genai
from google.genai import types

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

gemini_client = genai.Client(api_key="")


def wait_for_gemini_batch(
    job_name: str, folder: str, poll_interval: int = 300, max_wait_hours: int = 24
) -> None:
    """
    Wait for Gemini batch job to finish.
    """
    start = time.time()
    while True:
        batch_job = gemini_client.batches.get(name=job_name)
        state = batch_job.state.name if batch_job.state else None
        if state == "JOB_STATE_SUCCEEDED":
            get_batch_results_gemini(job_name, folder)
            break
        elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
            raise RuntimeError(f"Gemini job {job_name} failed with state {state}")
        elif (time.time() - start) > max_wait_hours * 3600:
            raise TimeoutError(f"Gemini job {job_name} exceeded {max_wait_hours}h")
        else:
            print(f"[Gemini] Job {job_name} still {state}... retry in {poll_interval}s")
            time.sleep(poll_interval)


def submit_batch_gemini(jsonl_file: str, model: str, display_name: str) -> str | None:
    """
    Submit a batch for gemini batch processing,
    return the job name.
    """
    try:
        uploaded_file = gemini_client.files.upload(
            file=jsonl_file,
            config=types.UploadFileConfig(
                display_name=display_name, mime_type="application/json"
            ),
        )
    except Exception:
        # Directly inspect the HTTP response if it exists
        traceback.print_exc()
        raise
    if uploaded_file.name:
        file_batch_job = gemini_client.batches.create(
            model=model,
            src=uploaded_file.name,
            config={
                "display_name": display_name,
            },
        )
        with open("data/temp_gemini_batch_id.txt", "w") as f:
            f.write(str(file_batch_job.name))
        return file_batch_job.name
    print("ERROR: no uploaded_file.name")


def get_batch_results_gemini(job_name: str, folder: str) -> None:
    """
    Retrieve the batch processing results from gemini,
    and save it into individual files.
    """
    if not job_name:
        with open("data/temp_gemini_batch_id.txt", "r") as f:
            job_name = f.read()
    batch_job = gemini_client.batches.get(name=job_name)
    if batch_job.error:
        print(f"Error: {batch_job.error}")

    state = batch_job.state.name if batch_job.state else None
    if state is None:
        print("batch_job.state is None")
        return
    if state != "JOB_STATE_SUCCEEDED":
        print(f"Batch {job_name} not ready: {state}")
        return
    # If batch job was created with a file
    if batch_job.dest and batch_job.dest.file_name:
        result_file_name = batch_job.dest.file_name
        file_content = gemini_client.files.download(file=result_file_name)

        with open(f"{folder}/gemini_results.txt", "wb") as f:
            f.write(file_content)
        with open(f"{folder}/gemini_results.txt", "r") as f:
            for line in f:
                data = json.loads(line)
                key = data["key"].split("/")[-1]
                response_text = data["response"]["candidates"][0]["content"]["parts"][
                    0
                ]["text"]
                model = data["response"]["modelVersion"]
                with open(f"{folder}/{key}_{model}.txt", "w") as f:
                    f.write(response_text)


if __name__ == "__main__":
    """CHANGE INSERTION MODEL"""
    model_family_insertion = "gemini"
    model_insertion = "gemini-2.5-pro"

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

    # ## ---------- Claim Extraction ---------- ###
    jsonl_file_gemini = f"{claim_folder}/{model_insertion}_claim_requests.jsonl"

    generate_claim_jsonl(
        model_family=model_family_insertion,
        jsonl_filename=jsonl_file_gemini,
        papers=papers,
        claim_folder=claim_folder,
        model=model_insertion,
    )

    if os.path.getsize(jsonl_file_gemini) != 0:
        job_name = submit_batch_gemini(
            jsonl_file_gemini, model_insertion, "error-request"
        )
        if job_name:
            run_stage_with_executor(
                wait_for_gemini_batch,
                claim_folder,
                job_name,
                "Claim Extraction",
            )

    ### ---------- Error Generation ---------- ###
    jsonl_file_gemini = f"{error_folder}/{model_insertion}_error_requests.jsonl"

    claims = generate_error_jsonl(
        model_family=model_family_insertion,
        jsonl_filename=jsonl_file_gemini,
        papers=papers,
        claim_folder=claim_folder,
        error_folder=error_folder,
        model=model_insertion,
    )

    if os.path.getsize(jsonl_file_gemini) != 0:
        split_files = split_jsonl_file(jsonl_file_gemini, lines_per_file=1000)

        for part_file in split_files:
            print(part_file)
            job_name = submit_batch_gemini(part_file, model_insertion, "error-request")
            if job_name:
                run_stage_with_executor(
                    wait_for_gemini_batch,
                    error_folder,
                    job_name,
                    f"Error Generation (batch part: {os.path.basename(part_file)})",
                )

    ### ---------- Error Filter ---------- ###
    jsonl_file_gemini = f"{filter_folder}/{model_insertion}_filter_requests.jsonl"

    generate_error_filter_jsonl(
        jsonl_filename=jsonl_file_gemini,
        filter_folder=filter_folder,
        error_folder=error_folder,
        model=model_insertion,
        model_family=model_family_insertion,
    )

    if os.path.getsize(jsonl_file_gemini) != 0:
        split_files = split_jsonl_file(jsonl_file_gemini, lines_per_file=1000)
        for part_file in split_files:
            print(part_file)
            job_name = submit_batch_gemini(part_file, model_insertion, "filter-request")
            if job_name:
                run_stage_with_executor(
                    wait_for_gemini_batch,
                    filter_folder,
                    job_name,
                    f"Error Filter (batch part: {os.path.basename(part_file)})",
                )
        valid_errors = get_valid_errors(error_folder, filter_folder, model_insertion)
    ### ---------- Error Location ---------- ###
    jsonl_file_gemini = f"{location_folder}/{model_insertion}_location_requests.jsonl"

    generate_error_location_jsonl(
        jsonl_filename=jsonl_file_gemini,
        error_folder=error_folder,
        filter_folder=filter_folder,
        location_folder=location_folder,
        model=model_insertion,
        model_family=model_family_insertion,
    )

    if os.path.getsize(jsonl_file_gemini) != 0:
        split_files = split_jsonl_file(jsonl_file_gemini, lines_per_file=1000)
        for part_file in split_files:
            print(part_file)
            job_name = submit_batch_gemini(
                part_file, model_insertion, "location-request"
            )
            if job_name:
                run_stage_with_executor(
                    wait_for_gemini_batch,
                    location_folder,
                    job_name,
                    f"Error Location (batch part: {os.path.basename(part_file)})",
                )

    ### ---------- Error Internal Identification ---------- ###
    create_altered_sources(
        error_folder=error_folder,
        filter_folder=filter_folder,
        model=model_insertion,
        hallucination_threshold=hallucination_threshold,
        version_control=version_control,
    )

    jsonl_file_gemini = (
        f"{identify_folder}/{model_insertion}_self_identify_request.jsonl"
    )

    generate_self_identification_jsonl(
        jsonl_filename=jsonl_file_gemini,
        error_folder=error_folder,
        location_folder=location_folder,
        identify_folder=identify_folder,
        filter_folder=filter_folder,
        model=model_insertion,
        model_family=model_family_insertion,
        version_control=version_control,
    )

    if os.path.getsize(jsonl_file_gemini) != 0:
        split_files = split_jsonl_file(jsonl_file_gemini, lines_per_file=1000)
        for part_file in split_files:
            print(part_file)
            job_name = submit_batch_gemini(
                part_file, model_insertion, "location-request"
            )
            if job_name:
                run_stage_with_executor(
                    wait_for_gemini_batch,
                    identify_folder,
                    job_name,
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
    print(remaining_errors)

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
