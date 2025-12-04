# FLAWS: Fault Localization Across Writing in Science

This repository provides the benchmark dataset, automated error insertion pipeline, and error identification & evaluation framework for the paper [FLAWS: A Benchmark for Error Identification and Localization in Scientific Papers](https://www.arxiv.org/abs/2511.21843). The code allows you to:

1. Evaluate LLMs on their ability to identify errors in scientific papers.  
2. Insert claim-specific errors into research papers using their LaTeX source to create examples similar to those in this dataset.

Set up the environment with:

```bash
cd FLAWS
conda env create -f environment.yml
conda activate llm_error
```

IMPORTANT: you'll also need **Ghostscript** (required for PDF compression) if it is not install on your system.
- Windows: download from [https://www.ghostscript.com/releases/gsdnld.html](https://www.ghostscript.com/releases/gsdnld.html)
- Linux: `sudo apt-get install ghostscript`
- Mac: `brew install ghostscript`

---

## A. Benchmark

The benchmark datasets are hosted on Hugging Face. Download and unzip them, and place the `ALL_GEMINI` and `ALL_OPENAI` folders in another folder `data` the same level as `src`:

- [FLAWS Dataset](https://huggingface.co/datasets/xasayi/FLAWS)

The current benchmark includes 713 papers:

- 265 unique papers, each with one error inserted using GPT 5  
- 448 unique papers, each with one error inserted using Gemini 2.5 Pro

Five frontier LLMs have been evaluated on this benchmark. Models are ranked using a logistic regression coefficient β_j, which measures performance in identifying errors. This ranking accounts for differences in errors inserted by the two insertion models, not just raw accuracy. For more details about how this score is calculated, refer to the [paper](https://www.arxiv.org/abs/2511.21843).

| **Identification Model** | **Rank** | **Score β_j** | **Accuracy @k=3** | **Accuracy @k=10** |
|--------------------------|----------|---------------|------------------|-------------------|
| GPT 5                    | 1        | 2.10          | 19.2%            | 39.1%             |
| Deepseek Reasoner v3.1   | 2        | 1.90          | 16.3%            | 35.2%             |
| Grok 4                   | 3        | 1.68          | 16.3%            | 23.4%             |
| Claude Sonnet 4.5        | 4        | 1.47          | 12.6%            | 21.5%             |
| Gemini 2.5 Pro           | 5        | 1.41          | 15.7%            | 19.8%             |

To evaluate other models by these 5 model providers - anthropic, deepseek, gemini, openai, xai - go to `src/evaluate_llm.py` and modify the following parameters to test different models:

```python
# CHANGE THE IDENTIFICATION MODELS
model_family_identification = "deepseek"  # Options: anthropic, deepseek, gemini, openai, xai
model_identification = "deepseek-reasoner"  # Specific model name
```

To run the evaluation code on all the paper-error pairs, use `python -m src.evaluate_llm`, this will produce a JSON results file and print out the identification accuracy. If you only wish to evaluate you model on a subset of paper-error pairs, modify this section:

```bash
with open(file, "r") as f:
    valid_errors = json.load(f)
identification_dict = defaultdict(list)

# if you only want to evaluate k examples, use replace with this line
# for paper, c in list(valid_errors.items())[:k]:
for paper, c in valid_errors.items():
    ...
```

To add a new model provider, update `src/utils/llm_calls.py`:

```python
def completion_response(model_family: str, model: str, prompt: str, pdf_path: str | None) -> str:
    completion_mapping = {
        "anthropic": get_completion_anthropic,
        "deepseek": get_completion_deepseek,
        "gemini": get_completion_gemini,
        "openai": get_completion_openai,
        "xai": get_completion_grok,
        ## create a new function and add it here
    }
    completion = completion_mapping[model_family]
    llm_output = completion(prompt=prompt, model=model, file=pdf_path)
    return llm_output
```

---

## B. Frameworks

To insert an error into a paper and evaluate an LLM's ability to identify it:

1. Place your LaTeX source folder in **data/papers/**  
2. Add your API keys to `src/utils/llm_calls.py` for single error insertion, or to the batch pipeline scripts for batch insertion:
   - `src/pipeline/batch_process/batching_pipeline_gemini.py`  
   - `src/pipeline/batch_process/batching_pipeline_openai.py`

### 1. Single Error Insertion

**Entry Point:**
To run the error insertion for a single paper, modify the following in `src/pipeline/single_process/single_error_insertion.py`:

```python
# Version control for results
version_control = "test"

# Specify paper and claim index
paper = "Matchmaker"  # LaTeX folder name in data/papers
ind = 0  # Claim index (0-indexed)
```

Then, use the following command

```bash
python -m src.pipeline.single_process.single_error_insertion
```


**Pipeline Steps:**
By running the insertion pipeline, it will perform the following steps sequentially:

1. Extract claims from the paper.
2. Generate an error for the specified claim.
3. Filter the error out if it is invalid or trivial.
4. If the error is not filtered out, insert it into the original LaTeX source.
5. Localize all of the text excerpts related to the error.
6. Perform self-identification using the same LLM to further filter out the error if it is too easy to identify.
7. Compile a modified PDF if the error is not filtered out.

### 2. Single Error Identification (+ Evaluation)

**Entry Point:**
To run the error identification and evaluation for a single paper-error pair, modify the following in `src/pipeline/single_process/single_error_identification.py`:

```python
# Ensure these variable names match the insertion pipeline
# Version control for results
version_control = "test"

# Specify papers and claim indices
papers = ["Matchmaker"]  # List of LaTeX folders
ind = [0]  # Corresponding claim indices
```

Then, you can run:

```bash
python -m src.pipeline.single_process.single_error_identification
```

### 3. Batch Processing

The previous modules process one paper-error pair at a time, but insertion can be batched using the following modules:

```bash
python -m src.pipeline.batch_process.batching_pipeline_gemini
python -m src.pipeline.batch_process.batching_pipeline_openai
```

Similarly, before running you may modify the following:

```python
# Version control for batch results
version_control = "test_batch"

# Papers to process
papers = get_papers("data/papers/")  # Returns all folder names in data/papers
```

The batch pipelines also run identification and evaluation after insertion.

---

## C. Other

To download papers from a specific conference using OpenReview, edit and then run `src/download_papers.py`:

```python
# OpenReview credentials
username = ""
password = ""

# Venue ID (e.g., ICML 2025)
venue_id = "ICML.cc/2025/Conference"
```