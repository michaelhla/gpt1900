"""
MATH dataset loader for RL training.
https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval

Uses SymPy-based symbolic equivalence from yale_physics for answer verification.
"""

import re
from datasets import load_dataset
from tasks.common import Task
from tasks.yale_physics import extract_answer_latex, is_symbolically_equivalent


# ---------------------------------------------------------------------------
# Extract gold answer from \boxed{...} in MATH solutions
# ---------------------------------------------------------------------------

def _extract_braced(text: str, start_pos: int) -> str | None:
    """Extract content from nested braces starting at start_pos (after opening brace)."""
    depth = 1
    i = start_pos
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start_pos : i - 1].strip()


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the answer from \\boxed{...} in a MATH solution string.

    Handles nested braces (e.g. \\boxed{\\frac{1}{2}}).
    Returns the last \\boxed{} match if multiple exist.
    """
    # Find all \boxed{ occurrences, return the last one (final answer)
    result = None
    for match in re.finditer(r"\\boxed\s*\{", solution):
        extracted = _extract_braced(solution, match.end())
        if extracted is not None:
            result = extracted
    return result


# ---------------------------------------------------------------------------
# MATH Task class
# ---------------------------------------------------------------------------

MATH_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
MATH_TYPES = [
    "Algebra", "Counting & Probability", "Geometry",
    "Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus",
]


class MATH(Task):
    """MATH benchmark dataset (DigitalLearningGmbH/MATH-lighteval).

    Args:
        split: "train" or "test"
        levels: optional list of difficulty levels to include (e.g. ["Level 1", "Level 2"])
        types: optional list of problem types to include (e.g. ["Algebra", "Geometry"])
    """

    def __init__(self, split, levels=None, types=None, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "MATH split must be train|test"
        ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)

        # Filter by level and type if specified
        if levels is not None:
            ds = ds.filter(lambda x: x["level"] in levels)
        if types is not None:
            ds = ds.filter(lambda x: x["type"] in types)

        self.ds = ds.shuffle(seed=42)

        # Pre-extract gold answers
        self.gold_answers = []
        for row in self.ds:
            answer = extract_boxed_answer(row["solution"])
            self.gold_answers.append(answer)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_question(self, index):
        """Return the problem statement string."""
        return self.ds[index]["problem"]

    def get_gold_answer(self, index):
        """Return the extracted gold answer string (from \\boxed{})."""
        return self.gold_answers[index]

    def get_metadata(self, index):
        """Return level and type metadata for a problem."""
        row = self.ds[index]
        return {"level": row["level"], "type": row["type"]}

    def get_example(self, index):
        """Return a conversation dict for SFT/eval compatibility."""
        row = self.ds[index]
        messages = [
            {"role": "user", "content": row["problem"]},
            {"role": "assistant", "content": row["solution"]},
        ]
        return {"messages": messages}

    def check_answer(self, index, response: str) -> bool:
        """Check if a response's \\answer{} or \\boxed{} matches the gold answer."""
        gold = self.gold_answers[index]
        if gold is None:
            return False
        pred = extract_answer_latex(response)
        if pred is None:
            return False
        result = is_symbolically_equivalent(pred, gold)
        return result is True
