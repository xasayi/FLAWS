# FLAWS: Fault Localization Across Writing in Science
This repository holds the current benchmark dataset as well as the automated error insertion pipeline + the automated error identification & evaluation framework for the paper [FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers]() (link to be added). Code here allows you to (i) evaluate LLMs on their ability to identify errors in scientific papers, (ii) insert claim-specific errors to research papers given their source LaTeX folders to create the dataset. 

Run the following commands to setup the environment:\
`conda env create -f environment.yml`\
`conda activate llm_error`

## A. Benchmark

The benchmark datasets used in this project are hosted on Hugging Face. Download and unzip them. Save the folder `FLAWS` at the same level as `src`:

- [FLAWS Dataset](https://huggingface.co/datasets/xasayi/FLAWS)

Currently, 5 frontier LLMs have been evaluated on this benchmark, with the following ranking:

| **Model** | **Ranking** | **Identification Accuracy**
|--------------------------|-------------|-------------|
| GPT 5                    | 1           |39.1%        |
| Deepseek Reasoner v3.1   | 2           |35.2%        |
| Grok 4                   | 3           |23.4%        |
| Claude Sonnet 4.5        | 4           |21.5%        |
| Gemini 2.5 Pro           | 5           |19.8%        |

The current benchmark consists of 713 papers, with 265 unique papers that have been each inserted with one error using gpt-5, and 448 unique papers that have been each inserted with one error using gemini-2.5-pro. To access the error identification ability of other models, you may run the module by using\
`python -m src.evaluate_llm`

Change the following arguments if you wish to test models other than the above providers
```python
"""CHANGE THE IDENTIFICATION MODELS"""
model_family_identification = "deepseek" # there is existing framework for deepseek, gpt, grok, gemini, claude
model_identification = "deepseek-reasoner" # specific model name
```

If you wish to evaluate models from other provides, then you may add code in `src.utils.llm_calls`. Include a function for calling the provider api, and then add a key-value pair in the dictionary in this function:

```python
def completion_response(
    model_family: str, model: str, prompt: str, pdf_path: str | None
) -> str:
    completion_mapping = {
        "openai": get_completion_openai,
        "gemini": get_completion_gemini,
        "anthropic": get_completion_anthropic,
        "grok": get_completion_grok,
        "deepseek": get_completion_deepseek,
        ### add new funciton here
    }
    completion = completion_mapping[model_family]
    llm_output = completion(prompt=prompt, model=model, file=pdf_path)
    return llm_output
```

---

## B. Frameworks
To insert an error into a specific paper and evaluate your LLM on how good it is at identify the inserted error, add your LaTeX source folder in **data/papers/**. \
To make API calls, add you API keys into the files `src/utils/llm_calls.py` for single error insertion. For batch insertion, add you API keys into either `src/pipeline/batch_process/batching_pipeline_gemini.py` or `src/pipeline/batch_process/batching_pipeline_openai.py`.

### 1. Error Insertion Pipeline
#### Entry Point for automated error insertion

`python -m src.pipeline.single_process.single_error_insertion`\
Running the above command will allow you to go through the entire error insertion pipeline for a single paper and claim in the paper. **BEFORE** you run this command, change the following things in the file (you can leave other settings as the default setting):

```python
...
"""CHANGE THIS FOR VERSION CONTROL"""
version_control = "cleanup" # the folder name of your results
...
"""CHANGE THESE FOR SPECIFIC PAPER AND CLAIM"""
paper = "Matchmaker" # the name of your LaTeX folder in data/papers
ind = 0 # the index of claims generated, it is 0-indexed
```

#### What This Script Does
When you run this module, it will go through the steps of 
1. Claim extraction from the paper.
2. Error generation based on the paper and extracted claim.
3. Filtering steps to discard easy/invalid errors.
4. Inserting the error into the latex source.
5. Localizing any text related to the error in the original latex source.
6. Self-identification of the error using the same LLM as the insertion model to filter out trivial errors.
7. Compile the modified pdfs of the remaining errors.

---

### 2. Error Evaluation Framework
#### Entry Point for automated error evalation
`python -m src.pipeline.single_process.single_error_insertion`\
Running the above command will allow you to go through the entire error identification and evaluation pipeline. **BEFORE** you run this command, change the following things in the file (you can leave other settings as the default setting):

```python
...
"""CHANGE THIS FOR VERSION CONTROL"""
version_control = "cleanup" # the folder name of your results
...
"""CHANGE THESE FOR SPECIFIC PAPER AND CLAIM"""
papers = ["Matchmaker"] # a list of folder names of the papers
ind = [1] # a list of the corresponding claim index used to generated the errors
```

---

### 3. Batch Processing
The error identification and evaluation only processed one paper-error pair at a time. However, the insertion process is batched using either the gemini batch API or the openai batch API. You may use

`python -m src/pipeline/batch_process/batching_pipeline_gemini` and\
`python -m src/pipeline/batch_process/batching_pipeline_openai`

to run the batches pipelines. They will also run the identification and evaluation pipeline at the end of the batched insertion process. **BEFORE** you run this command, change the following to specify your results folder and what papers you are running:

```python
...
"""CHANGE THIS FOR VERSION CONTROL"""
version_control = "cleanup_batch" # output folder name
...
"""CHANGE PAPER FORLDER AND EVALUATION PARAMETERS"""
papers = get_papers("data/papers/") # this function would return you all the folder names in data/papers
```

---

## C. Other

To download papers from a specific conference that has used OpenReview to perform peer review, refer to more details in the file `src.download_papers`. In particular, change the following

```python
# Enter your password and username for openreview
password = ""
username = ""
# change this to whichever venue you want to get papers from
venue_id = "ICML.cc/2025/Conference"
```