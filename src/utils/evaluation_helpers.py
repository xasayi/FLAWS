import re
import ast
import time
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate Levenshtein similarity between two sequences,
    return the similarity score.
    """
    tokens1 = s1.lower().split()
    tokens2 = s2.lower().split()
    m, n = len(tokens1), len(tokens2)
    if m == 0 and n == 0:
        return 1.0
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    ed_mat = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if tokens1[i - 1] == tokens2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # deletion
                dp[i][j - 1] + 1,  # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
            ed_mat[i][j] = 1 - dp[i][j] / max(i, j)
    # the larger score is, the more edits, less similar
    score = max(ed_mat[-1][:])
    return score


def best_subspan_similarity(pred: str, truth_spans: list[str], thres: float) -> float:
    """
    Find the best score between contiguous sentence subsets of pred
    and each item in truth_spans, return the highest score.
    """
    pred = pred.strip().lower()
    if not pred:
        return 0.0

    best_score = 0.0
    for truth in truth_spans:
        sentences = re.split(r"(?<=[.!?])\s+", truth)
        n = len(sentences)
        # generate subspans that covers different sentences
        for i in range(n):
            subspan = " ".join(sentences[i:])
            sim = levenshtein_similarity(pred, subspan)
            if sim > best_score:
                best_score = sim
                if best_score >= thres:
                    return best_score
    return best_score


def run_with_timeout(
    func: Callable, *args: Any, timeout: int | None = None, **kwargs: Any
) -> Any:
    """
    Run a function with a timeout in seconds,
    return function outputs.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout)


def calculate_levenshtein(
    folder: str,
    paper: str,
    ind: str | int,
    model: str,
    true_error_list: list[str],
    pred_error_list: list[str],
    threshold: float,
    top_k: int,
) -> float:
    """
    Find the score for the levenshtein metric,
    return the metric score.
    """
    name = f"{folder}/{paper}_{ind}_{model}_score.txt"
    pred_error_list = pred_error_list[:top_k]
    with open(name, "w") as f:
        filename = f.name
        scores1 = [
            best_subspan_similarity(pred, true_error_list, thres=threshold)
            for pred in pred_error_list
        ]
        f.write("subsets of ground vs pred:\n" + str(scores1) + "\n")
        if len(scores1) == 0:
            return 1
        if max(scores1) < threshold:
            scores2 = [
                best_subspan_similarity(ground, pred_error_list, thres=threshold)
                for ground in true_error_list
            ]
            f.write("\nsubsets of pred vs ground:\n" + str(scores2) + "\n")
            print(f"Saved internal score at {filename}")
            return max(scores2)
    print(f"Saved score at {filename}")
    return max(scores1)


def levenshtein_identify(
    folder: str,
    paper: str,
    ind: str | int,
    model: str,
    true_error_list: list[str],
    pred_error_list: list[str],
    threshold: float,
    top_k: int,
) -> bool:
    """
    Get the Levenshtein-based similarity,
    return whether the error was identified or not.
    """
    regeneration = True
    try:
        start = time.time()
        max_score = run_with_timeout(
            lambda: calculate_levenshtein(
                folder=folder,
                paper=paper,
                ind=ind,
                model=model,
                true_error_list=true_error_list,
                pred_error_list=pred_error_list,
                threshold=threshold,
                top_k=top_k,
            ),
            timeout=300,
        )
        end = time.time()
        print(
            f"Word-level Levenshtein distance-based similarity ran in {round(end - start, 2)} minutes"
        )

        if max_score < threshold:
            regeneration = False
        return regeneration
    except TimeoutError:
        print("Function call timed out!")
        return regeneration


def parse_llm_as_a_judge(text: str, k: int) -> bool:
    """
    Parse responses from using LLM as a judge to evaluated the identification,
    return identification results.
    """
    matching_text = 0
    correct_pattern = r"CORRECTLY IDENTIFIED"
    cor_matches = text.split(correct_pattern)
    text = correct_pattern.join(
        cor_matches[: min(k, len(cor_matches) - 1)] + [cor_matches[-1]]
    )

    pattern = r"(?<!IN)CORRECTLY IDENTIFIED"
    matches = re.findall(pattern, text)

    if matches:
        matching_text = len(matches)
    return matching_text != 0


def parse_levenshtein(text: str, k: int, threshold: float) -> int:
    """
    Parse results file from using levenshtein-based similarity,
    return identification result: 1(identified), 0(not identified).
    """
    subset_pattern = r"subsets of ground vs pred:\s*(\[[^\]]*\])"
    subset_match = re.search(subset_pattern, text)
    if subset_match:
        subset_list = ast.literal_eval(subset_match.group(1))[:k]
        subset_max_1 = max(subset_list)
    else:
        subset_max_1 = 0

    subset_pattern = r"subsets of pred vs ground:\s*(\[[^\]]*\])"
    subset_match = re.search(subset_pattern, text)
    if subset_match:
        subset_list = ast.literal_eval(subset_match.group(1))[:k]
        subset_max_2 = max(subset_list)
    else:
        subset_max_2 = 0
    return int(max(subset_max_1, subset_max_2) >= threshold)
