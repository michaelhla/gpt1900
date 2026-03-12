"""
Yale NLP PHYSICS benchmark reward function.

SymPy-based symbolic equivalence checking for LaTeX math answers.
No LLM fallback — deterministic, free, instant rewards.

Reference: https://github.com/yale-nlp/PHYSICS
"""

import re
import signal
from sympy import simplify, expand, trigsimp
from sympy.parsing.latex import parse_latex


# ---------------------------------------------------------------------------
# Answer extraction from \answer{...} with nested brace support
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


def extract_answer_latex(response: str) -> str | None:
    """Extract content from \\answer{...}, handling nested braces.

    Falls back to \\boxed{} if \\answer{} not found.
    """
    # Try \answer{...} first
    match = re.search(r"\\answer\s*\{", response)
    if match is not None:
        return _extract_braced(response, match.end())

    # Fallback: \boxed{...}
    match = re.search(r"\\boxed\s*\{", response)
    if match is not None:
        return _extract_braced(response, match.end())

    return None


# ---------------------------------------------------------------------------
# LaTeX preprocessing (adapted from Yale's equation_equivalency.py)
# ---------------------------------------------------------------------------

def _preprocess_latex(s: str) -> str:
    """Normalize LaTeX for SymPy parsing."""
    if not s:
        return ""
    # \dfrac → \frac (display variant, identical semantically)
    s = s.replace(r"\dfrac", r"\frac")
    s = s.replace(r"\tfrac", r"\frac")
    # Remove \displaystyle
    s = s.replace(r"\displaystyle", "")
    # Remove \text{...} and \mathrm{...} entirely (including braced content)
    s = re.sub(r"\\text\s*\{[^}]*\}", "", s)
    s = re.sub(r"\\mathrm\s*\{([^}]*)\}", r"\1", s)  # keep inner content for \mathrm
    s = re.sub(r"\\mbox\s*\{[^}]*\}", "", s)
    # Remove trailing text words (units/descriptions after the number)
    s = re.sub(r"(?<=\d)\s+[a-zA-Z][a-zA-Z\s]*$", "", s)
    # Remove degree symbol
    s = s.replace("°", "")
    s = s.replace(r"\degree", "")
    s = s.replace(r"\circ", "")
    # Remove thin/medium/thick spaces
    s = s.replace(r"\,", " ")
    s = s.replace(r"\;", " ")
    s = s.replace(r"\:", " ")
    s = s.replace(r"\!", "")
    s = s.replace(r"\ ", " ")
    # Remove commas from numbers (e.g., 430,080,000 → 430080000)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)
    s = re.sub(r"(\d),(\d)", r"\1\2", s)  # repeat for consecutive commas
    # Remove subscripts (variable decoration, not semantically meaningful for equivalence)
    s = re.sub(r"_\{.*?\}", "", s)
    s = re.sub(r"_\\?\w", "", s)
    # Remove formatting commands
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\cdot", "*")
    s = s.replace(r"\times", "*")
    # Extract RHS if equation (e.g. "x = \sqrt{2}")
    if r"\implies" in s:
        s = s.split(r"\implies")[-1].strip()
    if "=" in s:
        s = s.split("=")[-1].strip()
    # Clean up extra whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Symbolic equivalence via SymPy
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("SymPy computation timed out")


def _latex_to_float(s: str) -> float | None:
    """Try to convert a preprocessed LaTeX string to a float."""
    s = s.strip()
    if not s:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        pass
    # Try \frac{a}{b} pattern
    m = re.match(r"^-?\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}$", s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    # Try a/b pattern
    if "/" in s and "\\" not in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                pass
    return None


def _try_numeric_equal(a: str, b: str, rel_tol: float = 1e-6) -> bool | None:
    """Try to compare two LaTeX strings as floats."""
    fa = _latex_to_float(a)
    fb = _latex_to_float(b)
    if fa is None or fb is None:
        return None
    if fb == 0:
        return fa == 0
    return abs(fa - fb) / max(abs(fb), 1e-15) < rel_tol


def is_symbolically_equivalent(pred: str, gold: str, timeout: int = 10) -> bool | None:
    """
    Check if two LaTeX expressions are symbolically equivalent.

    Returns:
        True  — expressions are equivalent
        False — expressions are definitively not equivalent
        None  — could not parse one or both expressions
    """
    # 1. String normalization + exact match
    pred_clean = pred.strip().replace(" ", "")
    gold_clean = gold.strip().replace(" ", "")
    if pred_clean == gold_clean:
        return True

    # 2. Preprocess both and check again
    pred_pp = _preprocess_latex(pred)
    gold_pp = _preprocess_latex(gold)

    if pred_pp.replace(" ", "") == gold_pp.replace(" ", ""):
        return True

    # 3. Quick numeric comparison (catches 52.5 vs 105/2, commas, etc.)
    num_result = _try_numeric_equal(pred_pp, gold_pp)
    if num_result is True:
        return True

    # 4. SymPy symbolic comparison
    try:
        has_alarm = hasattr(signal, "SIGALRM")
        if has_alarm:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout)

        try:
            pred_expr = parse_latex(pred_pp)
            gold_expr = parse_latex(gold_pp)
        except Exception:
            return None
        finally:
            if has_alarm:
                signal.alarm(0)

        if has_alarm:
            signal.alarm(timeout)
        try:
            diff = simplify(expand(trigsimp(pred_expr - gold_expr)))
            if diff == 0:
                return True
            # 5. Numerical evaluation fallback
            if pred_expr.equals(gold_expr):
                return True
            # 6. Try evaluating both to float and comparing
            try:
                pred_f = complex(pred_expr.evalf())
                gold_f = complex(gold_expr.evalf())
                if pred_f.imag == 0 and gold_f.imag == 0:
                    pf, gf = pred_f.real, gold_f.real
                    if gf == 0:
                        if pf == 0:
                            return True
                    elif abs(pf - gf) / max(abs(gf), 1e-15) < 1e-6:
                        return True
            except Exception:
                pass
            return False
        except TimeoutError:
            return None
        except Exception:
            return None
        finally:
            if has_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Reward computation
# ---------------------------------------------------------------------------

def compute_reward(response: str, gold_answers: list[str]) -> float:
    """
    Compute correctness reward for a response against gold answers.

    For single-answer problems: 0 or 1.
    For multi-answer problems: fraction of correct answers (partial credit).

    Returns 0.0 if no \\answer{} tag found or unparseable.
    """
    pred = extract_answer_latex(response)
    if pred is None:
        return 0.0

    if len(gold_answers) == 1:
        result = is_symbolically_equivalent(pred, gold_answers[0])
        return 1.0 if result is True else 0.0

    # Multi-answer: extract comma-separated or multiple \answer{} tags
    # For now, try matching the single extracted answer against each gold
    correct = 0
    for gold in gold_answers:
        result = is_symbolically_equivalent(pred, gold)
        if result is True:
            correct += 1
    return correct / len(gold_answers)
