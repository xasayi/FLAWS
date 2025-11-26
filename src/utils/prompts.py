def generate_claim_prompt():
    prompt = """I will provide you with the full LaTeX source of a research paper. Your task is to read the paper thoroughly and list out all major, unique, and falsifiable claims made in the paper. The claims should follow these rules:

1. Each claim pertains to the core scientific content of the paper and its contributions.
2. Each claim is unique. 
3. You should not have repetitive claims or claims that are too similar to each other.

Please number the claims and return the output in this exact format:

1. <description of first falsifiable claim>

2. <description of second falsifiable claim>

3. ...

(...there can be as many falsifiable claims as necessary...)"""
    return prompt


def generate_error_generation_prompt(claim):
    prompt = """I am developing a research benchmark purely for research purposes to evaluate how effectively language models (LLMs) can detect errors in academic papers. Your task is to modify sections of a research paper in a way that introduces a non-trivial, plausible conceptual or theoretical oversight, grounded in domain-specific knowledge.

This benchmark aims to test LLMs' capacity for deep understanding, contextual inference, and expert-level critique, rather than surface-level textual correction.

Task Instructions
You will be given:

1. The input LaTeX source of a paper.
2. A key claim made in the paper.

Your job is to introduce a sophisticated, realistic academic error that undermines or weakens the provided claim. Try to introduce an error that would be a core conceptual error. This error should be coherent throughout the text and should:

1. Be subtle but significant, and potentially span multiple sections if needed for coherence.
2. Avoid obvious mistakes (e.g., grammar, formatting, numerical typos, misused citations).
3. Be non-trivial to detect without domain expertise, ideally requiring graduate-level reasoning or deeper.
4. Be internally consistent and plausible—the modified paper should still appear well-reasoned and coherent to a casual or intermediate reader.
5. Not be simple or obvious where one sentence is the direct contradiction of its neighboring sentences. 
6. Be similar to a GENUINE, REALISTIC mistake made by the authors of the paper.
7. Not be easily identified by a master student who has not read the paper in depth.  

- Make sure that the error is identifiable through the text alone. 
- DO NOT state any inconsistencies or limitations of the original input LaTeX source. 
- DO NOT correct any mistakes in the paper. Your job is to INTRODUCE an error NOT correct an error.

Modification Format
For each change, return:

:original_text:
<EXACT LaTeX excerpt from the original source — copy-pasted with NO modifications>

:modified_text:
<Modified excerpt that is LaTeX compilable— almost identical to the original but with a plausible conceptual error inserted>

Repeat original/modified pairs as needed if the error spans multiple non-contiguous parts.

STRICT FORMATTING RULES FOR the LaTeX source original_text:

1. Use PRECISE copy and paste methods from the LaTeX input to confirm excerpts are completely identical to the original source. 
2. DO NOT ALTER OR OMIT any token, including '$', '{', '}', or spacing.
3. DO NOT use quotation marks around the error excerpts.
4. DO NOT use ellipses ('...') within a single excerpt. If the original error texts are non-consecutive, return them as separate excerpts.
5. DO NOT omit any emph{}, textbf{}, textit{} or other formatting that is in the original LaTeX text. 
6. DO NOT hallucinate additional text that is not in the LaTeX source. 
7. DO NOT ALTER OR OMIT any line breaks or new lines.
8. DO NOT interchange punctuations like '' and " etc.
9. The excerpt you output must be exactly matchable in the input LaTeX source!

:explanation:
A brief but clear explanation of what conceptual error was introduced and why it is wrong, especially from a theoretical or domain-specific perspective.

You are given the following claim: 
"""
    prompt += claim
    return prompt


def generate_filter_invalid_error_prompt(
    original_text, modified_text, broken_claims, explanation_list
):
    prompt = """I am developing a research benchmark to evaluate how effectively language models (LLMs) can detect errors in academic papers. I have modified a particular section in the given paper in a way that introduces a non-trivial, plausible conceptual or theoretical oversight, grounded in domain-specific knowledge.

To modify the section, I have taken a claim in the paper, and introduced a sophisticated, realistic academic error that undermines or weakens the provided claim. 

I am going to provide you with the following:
1. Claim (The claim from the paper that will be undermined or weakened through the introduced error)
2. Original text (Part of the original LaTeX source that was altered, i.e. such that error is introduced)
3. Modified text (Modified version of the original text to include the error)
4. Explanation (The explanation behind the introduced error)
5. Input LaTeX source of the paper

Your job is very simple. You need to ensure that the error in the paper I have introduced through the modified text satisfies the below three sanity checks.
(a) The Modified Text should not explicitly state that there is an error.
(b) The Modified Text should not just highlight an existing weakness, shortcoming, or error in the paper. 
(c) The Modified Text should actually INTRODUCE a non-trivial, plausible conceptual or theoretical oversight, grounded in domain-specific knowledge.

If the Modified Text satisfies the above three criteria, then simply output "No changes required" without any other padding text.

If the Modified Text does not satisfy all the above criteria or does not contain an error, then simply output "Filtering required" without any other padding text."""
    original = "\n".join(original_text)
    modified = "\n".join(modified_text)
    claim = "\n".join(broken_claims)
    explanation = "\n".join(explanation_list)

    claim_prompt_addition = "\n\nCLAIM:\n" + claim + "\n\n"
    original_prompt_addition = "\n\nORIGINAL TEXT:\n" + original + "\n\n"
    modified_prompt_addition = "\n\nMODIFIED TEXT:\n" + modified + "\n\n"
    explanation_prompt_addition = "\n\nEXPLANATION:\n" + explanation + "\n\n"

    prompt += (
        claim_prompt_addition
        + original_prompt_addition
        + modified_prompt_addition
        + explanation_prompt_addition
    )
    return prompt


def generate_filter_easy_error_prompt(
    original_text, modified_text, broken_claims, explanation_list
):
    prompt = """I am developing a research benchmark to evaluate how effectively language models (LLMs) can detect errors in academic papers. I have modified a particular section in the given paper in a way that introduces a non-trivial, plausible conceptual or theoretical oversight, grounded in domain-specific knowledge.

To modify the section, I have taken a claim in the paper, and introduced a sophisticated, realistic academic error that undermines or weakens the provided claim. 

I am going to provide you with the following:
1. Claim (The claim from the paper that will be undermined or weakened through the introduced error)
2. Original text (Part of the original LaTeX source that was altered, i.e. such that error is introduced)
3. Modified text (Modified version of the original text to include the error)
4. Explanation (The explanation behind the introduced error)
5. Input LaTeX source of the paper

Your job is very simple. You need to ensure that the error in the paper I have introduced through the modified text cannot be easily identified by an LLM if it is just fed the PDF of this paper and asked to review it. By "cannot be easily identified", I mean it should be identified on average less than 20% of the time.

If the error introduced IS easily identifiable, then simply output "Filtering required" without any other padding text.

If the error introduced is difficult to identify, then simply output "No changes required" without any other padding text."""
    original = "\n".join(original_text)
    modified = "\n".join(modified_text)
    claim = "\n".join(broken_claims)
    explanation = "\n".join(explanation_list)

    claim_prompt_addition = "\n\nCLAIM:\n" + claim + "\n\n"
    original_prompt_addition = "\n\nORIGINAL TEXT:\n" + original + "\n\n"
    modified_prompt_addition = "\n\nMODIFIED TEXT:\n" + modified + "\n\n"
    explanation_prompt_addition = "\n\nEXPLANATION:\n" + explanation + "\n\n"

    prompt += (
        claim_prompt_addition
        + original_prompt_addition
        + modified_prompt_addition
        + explanation_prompt_addition
    )
    return prompt


def generate_error_location_prompt(
    original_text, modified_text, broken_claims, explanation_list
):
    prompt = """I am developing a research benchmark to evaluate how effectively language models (LLMs) can detect errors in academic papers. I have modified a particular section in the given paper in a way that introduces a non-trivial, plausible conceptual or theoretical oversight, grounded in domain-specific knowledge.

To modify the section, I have taken a claim in the paper, and introduced a sophisticated, realistic academic error that undermines or weakens the provided claim. 

I am going to provide you with the following:
1. Claim (The claim from the paper that will be undermined or weakened through the introduced error)
2. Original text (Part of the original LaTeX source that was altered, i.e. such that error is introduced)
3. Modified text (Modified version of the original text to include the error)
4. Explanation (The explanation behind the introduced error)
5. Input LaTeX source of the paper

Your job is to analyze the way the provided section has been modified and then output the part of the input LaTeX source that is incorrect (contains an error) as a result of this modification. It may be natural to think that the "original text" itself is the part of the paper that contains the error. While this may be the case, it may also happen that the original text in itself is correct but causes one or more other parts of the paper (input LaTeX source) to be incorrect (contains an error). The explanation just explains what the introduced error is. It is NOT the part of the input LaTeX source that contains the error.

Output ONLY the EXACT LaTeX excerpt from the input LaTeX source that is incorrect — copy-pasted with NO modifications, and do not output any padding text. START printing the LaTeX excerpts by printing "error:" followed by the incorrect excerpt(s).

STRICT FORMATTING RULES FOR the input LaTeX source:

1. Use PRECISE copy and paste methods from the LaTeX input to confirm excerpts are completely identical to the input LaTeX source. 
2. DO NOT ALTER OR OMIT any token, including '$', '{', '}', or spacing.
3. DO NOT use quotation marks around the error excerpts.
4. DO NOT use ellipses ('...') within a single excerpt. If the LaTeX excerpts are non-consecutive, return them as separate excerpts.
5. DO NOT omit any emph{}, textbf{}, textit{} or other formatting that is in the original LaTeX text. 
6. DO NOT hallucinate additional text that is not in the LaTeX source. 

Additionally, BEFORE you output the LaTeX excerpt in the above format, ABOVE that, I want the category of error that you have identified. 
Here are all the possible categories of errors:
1. Error in algorithm/proof
2. Error in reported results (e.g., compared to another paper or original paper)
3. Error in implementation (e.g., wrong concept/algorithm used, mismatch between theory and experiments or training and inference, plotting procedure, training procedure)
4. Inconsistencies in definitions
5. Error in assumptions (core conceptual)
6. Incorrect or incomplete analysis"""
    original = "\n".join(original_text)
    modified = "\n".join(modified_text)
    claim = "\n".join(broken_claims)
    explanation = "\n".join(explanation_list)

    claim_prompt_addition = "\n\nCLAIM:\n" + claim + "\n\n"
    original_prompt_addition = "\n\nORIGINAL TEXT:\n" + original + "\n\n"
    modified_prompt_addition = "\n\nMODIFIED TEXT:\n" + modified + "\n\n"
    explanation_prompt_addition = "\n\nEXPLANATION:\n" + explanation + "\n\n"

    prompt += (
        claim_prompt_addition
        + original_prompt_addition
        + modified_prompt_addition
        + explanation_prompt_addition
    )
    return prompt


def generate_internal_error_identification_prompt(num_chunks=5, word_limit=100):
    prompt = f"""Attached is the LaTeX source of a research paper. I have taken a claim in the paper and tried to break it by modifying one or more snippets in the original LaTeX source. You have been provided the modified LaTeX source. Your task is to identify error text chunks that pertain to the error that breaks a claim in the paper, and return them in ranked order where the first returned error text is the most serious. The identified error text should be a substantial technical flaw that could undermine the paper's validity and potentially lead to rejection from a scientific conference or journal.

Important constraints:
- Do not identify minor issues such as grammar, style, formatting, or typographical errors.
- Only use knowledge available up to the time of the paper's publication. Do not search the web or use
information published afterward.
- Return exact excerpts from the paper (do not paraphrase).
- Each excerpt must be at most {word_limit} words long.
- Return at most {num_chunks} error chunks, ranked in order of seriousness (most serious first).

Output format (strictly follow):

:error_text:
<exact excerpt from the source paper>

:error_text:
<exact excerpt from the source paper>

(...there can be a varying number of error texts for each error, the total number of error texts should be at most {num_chunks})

:explanation:
<clear and precise explanation of the error>"""
    return prompt


def generate_error_identification_prompt(num_chunks=10, word_limit=100):
    prompt = f"""Attached is the PDF of a research paper. Your task is to identify error text chunks that pertain to the MOST serious errors in the paper, and return them in ranked order where the first returned error text is the most serious. These should be substantial technical flaws that could undermine the paper's validity and potentially lead to rejection from a scientific conference or journal.

Important constraints:
- Do not identify minor issues such as grammar, style, formatting, or typographical errors.
- Only use knowledge available up to the time of the paper's publication. Do not search the web or use
information published afterward.
- Return exact excerpts from the paper (do not paraphrase).
- Each excerpt must be at most {word_limit} words long.
- Return at most {num_chunks} error chunks, ranked in order of seriousness (most serious first).

Output format (strictly follow):

:error_text:
<exact excerpt from the source paper>

:error_text:
<exact excerpt from the source paper>

(...there can be a varying number of error texts for each error, the total number of error texts should be at most {num_chunks})

:explanation:
<clear and precise explanation of the error>"""
    return prompt


def generate_error_evaluation_prompt(true_error_text, pred_error_text):
    prompt = f"""I have created a machine learning model to identify errors in a research paper. It identifies the error by pointing to a specific excerpt of the research paper. The output of the model I am going to provide you with is the list of identified errors.

I also have some ground truth errors. These are the actual errors in the research paper. For this, I am going to provide you with a list of ground truth errors which are again excerpts from the research paper.

Note that the model need not output errors in the same order as the ground truth errors. Your job is to check each of the identified errors, and see if there is a corresponding ground truth error. For an accurate identified error, at least one ground truth error must be from the same excerpt as the identified error (i.e., there must be at least one match).

Criteria for a match:
The excerpts of the identified and ground truth errors should be evident that they are the same excerpt from the paper. It is possible that one may be in LaTeX source code and one is plain text, etc. Either the identified error must be a subset of at least one of the ground truth errors OR at least one of the ground truth errors must be a subset of the identified error.

Provide the output in the following manner:
For each identified error, provide the identified error and then add "CORRECTLY IDENTIFIED" if it is a correctly identified error (i.e., there is a match with at least one ground truth error), or add "INCORRECTLY IDENTIFIED" if there is not match with any ground truth error.

DO NOT EVALUATE the correctness of the error on your own. The correctness is purely determined based on the given ground truth errors. They HAVE to be the same excerpt of the paper.

IDENTIFIED ERRORS:
{pred_error_text}

ACTUAL ERRORS:
{true_error_text}"""
    return prompt
