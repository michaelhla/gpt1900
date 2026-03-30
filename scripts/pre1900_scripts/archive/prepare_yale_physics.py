#!/usr/bin/env python3
"""
Download and filter Yale NLP PHYSICS benchmark for pre-1900 RL training.

Downloads 4 text-only domain JSONL files (mechanics, electro, optics, statistics),
applies anachronism keyword filtering, splits single vs multi-answer, and outputs
parallel JSONL files matching the existing RL data format.

Usage:
    python -m scripts.pre1900_scripts.prepare_yale_physics [--base-dir $NANOCHAT_BASE_DIR]
"""

import argparse
import json
import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

from scripts.pre1900_scripts.constants import QUANTITATIVE_REASONING_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Domains to use (skip atomic & quantum — entirely post-1900)
# ---------------------------------------------------------------------------
DOMAINS = ["mechanics", "electro", "optics", "statistics"]

GITHUB_RAW_BASE = (
    "https://raw.githubusercontent.com/yale-nlp/PHYSICS/main/"
    "PHYSICS/PHYSICS-textonly"
)

# ---------------------------------------------------------------------------
# Anachronism filter patterns (subset of clean_year_split.py ALWAYS_REJECT_PATTERNS)
# Tuned for filtering *physics problems* rather than full documents.
# We use a compact subset focused on concepts that appear in problem text.
# ---------------------------------------------------------------------------
ANACHRONISM_PATTERNS = [
    # Particles & quantum objects
    r"\bphoton(?:s|ic)?\b",
    r"\bpositron(?:s)?\b",
    r"\bneutrino(?:s|o)?\b",
    r"\bmuon(?:s)?\b",
    r"\bgluon(?:s)?\b",
    r"\bmeson(?:s)?\b",
    r"\bfermion(?:s)?\b",
    r"\bboson(?:s)?\b",
    r"\bhadron(?:s|ic)?\b",
    r"\bbaryon(?:s|ic)?\b",

    # Quantum mechanics
    r"\bquantum\b",
    r"\bquantized\b",
    r"\bquantization\b",
    r"\bwave\s*[\s-]?function(?:s)?\b",
    r"\bwavefunction\b",
    r"\buncertainty\s+principle\b",
    r"\bexclusion\s+principle\b",
    r"\bpauli\b",
    r"\bde\s+broglie\b",
    r"\bcompton\s+(?:scattering|effect|wavelength)\b",
    r"\bdirac\b",
    r"\bfeynman\b",
    r"\bpath\s+integral\b",
    r"\bspin[- ]orbit\s+coupling\b",
    r"\bzero[- ]point\s+energy\b",
    r"\bplanck(?:'?s)?\s+(?:constant|law|distribution|spectrum|radiation|formula)\b",
    r"\b[Ss]-wave\b",                               # quantum scattering (partial waves)
    r"\bspin\s+\$?\s*1/2\b",                        # quantum spin (1925), handles "spin $ 1/2 $"
    r"\bspin\s+\$?\s*\d+/\d+\b",                   # any fractional spin
    r"\bspin\s+(?:up|down)\b",                      # quantum spin states
    r"\bspins\b",                                   # plural "spins" = quantum spin
    r"\belectron(?:s)?\b",                          # electron as particle (Thomson 1897, Drude 1900+)
    r"\bfree\s+electron(?:s)?\b",                   # Drude model (1900) / quantum

    # Relativity
    r"\bspacetime\b",
    r"\bspace-time\b",
    r"\bspecial\s+relativity\b",
    r"\bgeneral\s+relativity\b",
    r"\brelativistic\b",
    r"\btime\s+dilation\b",
    r"\blength\s+contraction\b",
    r"\bmass[–-]energy\b",
    r"\be\s*=\s*mc\s*(?:\^|²|2)\b",
    r"\blorentz\s+(?:transformation|contraction|factor|invariance|covariance)\b",
    r"\bminkowski\b",
    r"\beinstein(?:'?s)?\b",
    r"\bphotoelectric\s+effect\b",
    r"\brest\s+(?:energy|mass)\b",                  # E=mc² concept (1905)
    r"m_\{?0\}?",                                   # rest mass m_0 in LaTeX
    r"\\gamma\s*[\^{]",                             # Lorentz factor γ² or γ^{n}
    r"\bgravitational\s+red\s*shift\b",             # GR (1915)
    r"\b[Rr]iemannian\b",                           # Riemannian metric = GR context
    r"g_\{?[0-3][0-3]\}?",                          # metric tensor components g_{00} etc.
    r"\bspace\s+travele?r\b",                       # SR thought experiments

    # Nuclear & particle physics
    r"\bnuclear\b",
    r"\batomic\s+number\b",
    r"\bisotope(?:s|ic)?\b",
    r"\bradioactive\b",
    r"\bhalf[- ]life\b",
    r"\bfission\b",
    r"\bfusion\b(?=.*\b(?:nuclear|reactor|plasma|thermonuclear)\b)",
    r"\bcosmic\s+ray(?:s)?\b",                      # Hess 1912
    r"\bproton(?:s)?\b",                             # Rutherford 1920

    # Condensed matter (post-1900)
    r"\bsuperconduct(?:ivity|ing|or)\b",
    r"\bsuperfluid\b",
    r"\bbose[- ]einstein\b",
    r"\bfermi[- ]dirac\b",
    r"\bfermi\s+(?:level|energy|surface|gas)\b",
    r"\bband\s+gap\b",
    r"\bphonon(?:s)?\b",
    r"\bplasmon(?:s)?\b",
    r"\bexciton(?:s)?\b",

    # Technology & post-1900 engineering
    r"\blaser(?:s)?\b",
    r"\btransistor(?:s)?\b",
    r"\bsemiconductor(?:s)?\b",
    r"\bLED\b",
    r"\bwaveguide(?:s)?\b",
    r"\bantenna(?:e|s)?\b",                         # antenna arrays (post-1900 radio engineering)
    r"\bradar\b",                                   # radar (1930s)
    r"\bspace\s+(?:telescope|shuttle|station)\b",   # space technology (post-1900)

    # Named physicists (unambiguous post-1900 work)
    r"\bschr[öo]dinger\b",
    r"\bheisenberg\b",
    r"\bbohr(?:'?s)?\s+(?:model|radius|atom|magneton)\b",
    r"\brutherford(?:'?s)?\s+(?:model|scattering)\b",

    # Cosmology
    r"\bbig\s+bang\b",
    r"\bdark\s+matter\b",
    r"\bdark\s+energy\b",
    r"\bblack\s+hole\b",
    r"\bneutron\s+star\b",

    # Debye model (1912)
    r"\bdebye\b",
]

_ANACHRONISM_RE = re.compile("|".join(ANACHRONISM_PATTERNS), re.IGNORECASE)


def is_anachronistic(text: str) -> tuple[bool, str | None]:
    """Check if text contains post-1900 physics concepts."""
    m = _ANACHRONISM_RE.search(text)
    if m:
        return True, m.group()
    return False, None


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def download_domain_file(domain: str, cache_dir: Path) -> Path:
    """Download a domain JSONL file from GitHub if not cached."""
    filename = f"{domain}_dataset_textonly.jsonl"
    cached = cache_dir / filename
    if cached.exists():
        return cached

    url = f"{GITHUB_RAW_BASE}/{filename}"
    print(f"  Downloading {url}...")
    # Try local clone first (faster if already cloned)
    local_clone = Path("/tmp/yale_physics/PHYSICS/PHYSICS-textonly") / filename
    if local_clone.exists():
        import shutil
        shutil.copy2(local_clone, cached)
        return cached

    subprocess.run(
        ["curl", "-sL", "-o", str(cached), url],
        check=True,
    )
    return cached


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_and_filter(domain: str, cache_dir: Path) -> list[dict]:
    """Load a domain file and filter out anachronistic problems."""
    filepath = download_domain_file(domain, cache_dir)
    kept = []
    rejected_reasons: dict[str, int] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Check questions + solutions + final_answers for anachronisms
            answers_text = " ".join(obj.get("final_answers", []))
            combined = (obj.get("questions", "") or "") + " " + (obj.get("solutions", "") or "") + " " + answers_text
            is_bad, matched = is_anachronistic(combined)
            if is_bad:
                bucket = matched[:30] if matched else "unknown"
                rejected_reasons[bucket] = rejected_reasons.get(bucket, 0) + 1
                continue

            # Skip problems with graphs (not text-only in practice)
            if obj.get("graphs"):
                rejected_reasons["has_graphs"] = rejected_reasons.get("has_graphs", 0) + 1
                continue

            # Must have at least one final answer
            answers = obj.get("final_answers", [])
            if not answers or all(not a.strip() for a in answers):
                rejected_reasons["no_answer"] = rejected_reasons.get("no_answer", 0) + 1
                continue

            obj["_domain"] = domain
            obj["_num_answers"] = len(answers)
            kept.append(obj)

    total = len(kept) + sum(rejected_reasons.values())
    print(f"  {domain}: {total} total -> {len(kept)} kept ({total - len(kept)} rejected)")
    if rejected_reasons:
        top = sorted(rejected_reasons.items(), key=lambda x: -x[1])[:5]
        for reason, count in top:
            print(f"    {reason}: {count}")

    return kept


def main():
    parser = argparse.ArgumentParser(description="Prepare Yale PHYSICS data for RL")
    parser.add_argument("--base-dir", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="/tmp/yale_physics_cache")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--single-answer-only", action="store_true", default=True,
                        help="Only keep single-answer problems (default: True)")
    parser.add_argument("--include-multi-answer", action="store_true",
                        help="Also include multi-answer problems")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.include_multi_answer:
        args.single_answer_only = False

    base_dir = args.base_dir or os.environ.get("NANOCHAT_BASE_DIR")
    if not base_dir:
        # Default to repo root
        base_dir = str(Path(__file__).resolve().parent.parent.parent)

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(base_dir) / "instruct_data" / "yale_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    # Load and filter all domains
    print("Loading and filtering Yale PHYSICS text-only problems...")
    all_problems = []
    for domain in DOMAINS:
        problems = load_and_filter(domain, cache_dir)
        all_problems.extend(problems)

    print(f"\nTotal after anachronism filter: {len(all_problems)}")

    # Filter to single-answer only if requested
    if args.single_answer_only:
        single = [p for p in all_problems if p["_num_answers"] == 1]
        multi = len(all_problems) - len(single)
        print(f"Single-answer only: {len(single)} kept, {multi} multi-answer excluded")
        all_problems = single

    # Stratified train/val split by domain
    by_domain: dict[str, list[dict]] = {}
    for p in all_problems:
        by_domain.setdefault(p["_domain"], []).append(p)

    train_problems = []
    val_problems = []
    for domain, problems in by_domain.items():
        random.shuffle(problems)
        n_val = max(1, int(len(problems) * args.val_fraction))
        val_problems.extend(problems[:n_val])
        train_problems.extend(problems[n_val:])

    random.shuffle(train_problems)
    random.shuffle(val_problems)

    print(f"\nSplit: {len(train_problems)} train, {len(val_problems)} val")
    for domain in DOMAINS:
        n_train = sum(1 for p in train_problems if p["_domain"] == domain)
        n_val = sum(1 for p in val_problems if p["_domain"] == domain)
        print(f"  {domain}: {n_train} train, {n_val} val")

    # Write output files
    system_message = {"role": "system", "content": QUANTITATIVE_REASONING_SYSTEM_PROMPT}

    def write_split(problems: list[dict], split: str):
        problems_path = output_dir / f"yale_problems_{split}.jsonl"
        prompts_path = output_dir / f"yale_prompts_sys_{split}.jsonl"

        with open(problems_path, "w", encoding="utf-8") as fp, \
             open(prompts_path, "w", encoding="utf-8") as fs:
            for p in problems:
                # Problems file: metadata + gold_answers array
                problem_obj = {
                    "id": p["id"],
                    "domain": p["_domain"],
                    "questions": p["questions"],
                    "gold_answers": p["final_answers"],
                }
                fp.write(json.dumps(problem_obj, ensure_ascii=False) + "\n")

                # Prompts file: CustomJSON format with system prompt
                messages = [
                    system_message,
                    {"role": "user", "content": p["questions"]},
                    {"role": "assistant", "content": "(to be generated)"},
                ]
                fs.write(json.dumps(messages, ensure_ascii=False) + "\n")

        print(f"  Wrote {len(problems)} problems to {problems_path.name}")
        print(f"  Wrote {len(problems)} prompts to {prompts_path.name}")

    write_split(train_problems, "train")
    write_split(val_problems, "val")

    # Verify: spot-check SymPy parse rate on gold answers
    print("\nVerifying SymPy parse rate on gold answers...")
    try:
        from sympy.parsing.latex import parse_latex as _parse
        parsed, failed, total = 0, 0, 0
        for p in all_problems:
            for ans in p["final_answers"]:
                total += 1
                try:
                    _parse(ans)
                    parsed += 1
                except Exception:
                    failed += 1
                    if failed <= 5:
                        print(f"    Parse fail: {ans!r} (id={p['id']})")
        print(f"  SymPy parse rate: {parsed}/{total} ({100*parsed/max(total,1):.1f}%)")
    except ImportError:
        print("  sympy not available, skipping parse rate check")

    print("\nDone.")


if __name__ == "__main__":
    main()
