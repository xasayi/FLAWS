import os
import difflib
from src.utils.formatting import format_generated_error, clean_latex_block


def find_best_fuzzy_window(
    full_source: str, excerpt: str, margin: int
) -> tuple[int, str, float]:
    """
    Find best fuzzy window in the full source that matches the excerpt,
    return the best starting index in the full source,
    the best match in the excerpt, and the matching ratio.
    """
    max_ratio = 0
    best_index = -1
    best_match = ""

    for i in range(0, len(full_source) - len(excerpt) + 1):
        window_start = max(0, i - margin)
        window_end = i + len(excerpt) + margin
        window = full_source[window_start:window_end]

        ratio = difflib.SequenceMatcher(None, window, excerpt).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            best_index = window_start
            best_match = window

    return best_index, best_match, max_ratio


def get_matched_span(full_source_segment: str, excerpt: str) -> tuple[int, int]:
    """
    Find best matching span source segment and excerpt,
    return the start and end indices.
    """
    matcher = difflib.SequenceMatcher(None, full_source_segment, excerpt)
    blocks = matcher.get_matching_blocks()
    start = min(block.a for block in blocks if block.size > 0)
    end = max(block.a + block.size for block in blocks if block.size > 0)
    return start, end


def replace_excerpt(
    full_source: str | None,
    excerpt: str,
    replacement: str,
    threshold: float,
    margin: int = 16,
) -> tuple[str | None, int | None, int | None]:
    """ "
    Replace the original with the modified text,
    return the source with replaced text and the actual start and end indices.
    """
    if full_source is None:
        return None, None, None
    start_index, best_window, ratio = find_best_fuzzy_window(
        full_source, excerpt, margin
    )
    print("Ratio between two excerpt:", ratio)
    if ratio < threshold:
        return None, None, None
    match_start, match_end = get_matched_span(best_window, excerpt)

    # Clip length to not exceed excerpt + margin
    match_len = match_end - match_start
    max_allowed_len = len(excerpt) + margin
    if match_len > max_allowed_len:
        match_end = match_start + max_allowed_len

    # Compute indices in original full source
    true_start = start_index + match_start
    true_end = start_index + match_end

    # Sanity check the end index
    if true_end > len(full_source):
        true_end = len(full_source)
    modified_source = full_source[:true_start] + replacement + full_source[true_end:]
    return modified_source, true_start, true_end


def modify_source(
    error_filename: str, latex_source: str, threshold: float
) -> str | None:
    """
    Modify the source to have the errors text inserted,
    return the modified latex source.
    """
    modified_text, original_text, _, _ = format_generated_error(error_filename)
    for i in range(len(original_text)):
        original = original_text[i].lstrip()
        original = clean_latex_block(original)
        modified = modified_text[i].lstrip()
        modified = clean_latex_block(modified)

        ori_start = latex_source.find(original)
        if ori_start == -1:
            print("No exact match found. Using difflib to match.")
            new_source, start, end = replace_excerpt(
                latex_source, original, modified, threshold=threshold
            )
            if new_source is None:
                print("No match found.")
                return None
            latex_source = new_source
        else:
            print("Exact match found.")
            latex_source = (
                latex_source[:ori_start]
                + modified
                + latex_source[ori_start + len(original) :]
            )
    return latex_source


def create_altered_latex(
    paper: str, new_latex: str, version: str, new_main_file: str
) -> str:
    """
    Create and save an altered latex tex file,
    return new latex filename.
    """
    folder = f"data/{version}/altered_papers/{paper}/"
    os.makedirs(folder, exist_ok=True)

    with open(f"{folder}/{new_main_file}.tex", "w") as f:
        f.write(new_latex)
        filename = f.name
    print(f"Created altered LaTeX file at {filename}.")
    return filename
