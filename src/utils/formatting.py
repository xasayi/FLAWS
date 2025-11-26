import re
from collections import defaultdict


def format_claims(filename: str) -> list[str]:
    """
    Format identified claims into a list.
    """
    with open(filename, "r") as f:
        content = f.read()
    parts = re.split(r"(?m)^\s*\d+\.\s*", content)
    claim_list = [p.strip() for p in parts[1:] if p.strip()]
    return claim_list


def format_generated_error(
    filename: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Format identified modified text, original text, claims, explanation into lists.
    """
    with open(filename, "r") as file:
        output = file.read()

    pattern = r":(explanation|claim|modified_text|original_text):\s*(.*?)(?=\n:\w+:|$)"
    matches = re.findall(pattern, output, re.DOTALL)

    sections = defaultdict(list)
    for keyword, content in matches:
        sections[keyword].append(content.strip())
    sections = dict(sections)

    return (
        sections.get("modified_text", [""]),
        sections.get("original_text", [""]),
        sections.get("explanation", [""]),
        sections.get("claim", [""]),
    )


def format_localized_error(filename: str) -> list[str]:
    """
    Format localized errors into a list.
    """
    with open(filename, "r") as f:
        output = f.read()
    return [i.strip() for i in output.split("error:")[1:]]


def format_identified_error(filename: str) -> list[str]:
    """
    Format identified errors into a list.
    """
    with open(filename, "r") as f:
        output = f.read()
    return [i.strip() for i in output.split("error_text:")[1:]]


def clean_latex_block(text: str) -> str:
    """Remove markdown style latex code fenses and extra
    formatting characters from a block of text, return cleaned text."""
    text = text.strip()
    if text.startswith("```latex"):
        text = text[len("```latex") :].lstrip()
    if text.endswith("```"):
        text = text[: -len("```")].rstrip()
    text = text.strip("-` \n")
    return text
