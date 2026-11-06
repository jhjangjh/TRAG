"""Answer parsing for the two-caption compositional-reasoning benchmarks.

The prompt asks for "(1)" or "(2)". A response counts only when it clearly
selects one caption, in one of three ways:

1. it states a caption index, e.g. "(2)", "2", "Answer: (1)";
2. it reproduces exactly one of the two captions (case, spacing and
   punctuation ignored);
3. it answers with the letter "A" or "B" for the first or second caption.

Anything else -- an empty response, other text, or an answer naming both
captions -- is unparseable and is scored as incorrect.
"""
import re


def _normalize(text):
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def parse_choice(output_text, option_1=None, option_2=None):
    """Return 1 or 2 if the response clearly selects that caption, else None."""
    text = output_text.strip().lower()

    # 1. Caption index.
    bracketed = set(re.findall(r"\(([12])\)", text))
    if len(bracketed) == 1:
        return int(bracketed.pop())
    if bracketed:
        return None  # e.g. "(1) or (2)"
    match = re.match(r"^(?:answer\s*:?\s*|option\s*|caption\s*)?([12])(?![0-9])", text)
    if match:
        return int(match.group(1))

    # 2. Verbatim copy of exactly one caption.
    if option_1 is not None and option_2 is not None:
        response, first, second = _normalize(text), _normalize(option_1), _normalize(option_2)
        if response and (response == first) != (response == second):
            return 1 if response == first else 2

    # 3. Letter answer.
    match = re.fullmatch(r"(?:answer\s*:?\s*)?(?:option\s*)?\(?([ab])\)?\.?", text)
    if match:
        return 1 if match.group(1) == "a" else 2
    return None
