#!/usr/bin/env python3
"""
Create a year-split dataset by combining pre-1900 data with a year-filtered
portion of the 1900-1914 supplement.

Two phases (can be run independently with --phase):
  Phase 1 (Dask-parallel): Filter the supplement by year (<= end_year) AND
      apply anachronism filter calibrated to the cutoff. Writes cleaned
      supplement shards to a staging directory.
  Phase 2 (serial): Reads ALL pre-1900 staged files PLUS the newly filtered
      supplement, chunks, balances, and writes final output shards.

The anachronism filter is calibrated per cutoff year:
  - Starts from the strict pre-1900 filter (clean_full_corpus.py)
  - Removes patterns for discoveries that occurred <= end_year
  - Date regex catches (end_year+1)+ instead of 1900+

Default: --end-year 1904 (pre-1905 split)
  Removed from pre-1900 filter (legitimate by 1904):
    planck constant/law/... (Planck 1900), quantum theory (1900),
    quanta of energy/light/radiation/action (1900),
    radioactivity (1898 Curie), radioactive decay (1902 Rutherford/Soddy)

Output format:
    shard_XXXXX.parquet  (text column only, zstd compression)
    Last shard = validation split

Usage:
    python scripts/pre1900_scripts/clean_year_split.py \
        --end-year 1904 \
        --pre1900-staging /mnt/main0/data/michaelhla/pre1900_full_clean/_staging \
        --supplement-raw /mnt/main0/data/michaelhla/pre1915_supplement_raw \
        --output /mnt/main0/data/michaelhla/pre1905_full_clean \
        --workers 64 --world-size 64
"""

from __future__ import annotations
import argparse
import gc
import os
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# Anachronism Filter for 1905 cutoff (end_year=1904)
# ============================================================================
# Starts from the strict pre-1900 filter and removes patterns for discoveries
# that occurred by 1904:
#
# Removed (legitimate by 1904):
#   planck constant/law/... (Planck 1900)
#   quantum theory (Planck 1900)
#   quanta of energy/light/radiation/action (Planck 1900)
#   radioactivity (1898 Curie)
#   radioactive decay (1902 Rutherford/Soddy)
#
# Still rejected (1905+):
#   einstein, special relativity, photoelectric effect (1905)
#   spacetime, space-time (1908 Minkowski)
#   relativistic, lorentz transformation/contraction/... (formalized 1905+)
#   minkowski space/... (1908)
#   time dilation, length contraction, mass-energy, e=mc² (1905)
#   isotope (1913 Soddy), atomic number (1913 Moseley)
#   bohr model/atom/... (1913), sommerfeld (1915+)
#   superconductivity (1911), geiger counter (1908), cloud chamber (1911)
#   main sequence (1911 Hertzsprung), hertzsprung-russell (1911-1913)
#   half-life (1907 Rutherford), sonar (1906)
# ============================================================================

ALWAYS_REJECT_PATTERNS = [
    # -- Particles & quantum objects (no pre-1904 meaning) --
    r"\bphoton(?:s|ic)?\b",                     # 1926 Lewis
    r"\bpositron(?:s)?\b",                       # 1932 Anderson
    r"\bneutrino(?:s|o)?\b",                     # 1930 Pauli / 1956 detected
    r"\bmuon(?:s)?\b",                           # 1936
    r"\bgluon(?:s)?\b",                          # 1979
    r"\bmeson(?:s)?\b",                          # 1947
    r"\bfermion(?:s)?\b",                        # named after Fermi
    r"\bkaon(?:s)?\b",                           # 1947
    r"\bpion(?:s)?\b",                           # 1947
    r"\btauon(?:s)?\b",
    r"\bhadron(?:s|ic)?\b",
    r"\bbaryon(?:s|ic)?\b",

    # -- Quantum mechanics terminology (post-1904 only) --
    r"\bquantum\s+mechan",                       # quantum mechanics/mechanical
    r"\bquantum\s+field",                        # quantum field theory
    r"\bquantum\s+electrodynamics\b",
    r"\bquantum\s+chromodynamics\b",
    # NOTE: quantum theory REMOVED (Planck 1900)
    r"\bquantum\s+physics\b",
    r"\bquantum\s+state(?:s)?\b",
    r"\bquantum\s+number(?:s)?\b",
    r"\bquantum\s+level(?:s)?\b",
    r"\bquantum\s+effect(?:s)?\b",
    r"\bquantum\s+entangle",
    r"\bquantum\s+superposition\b",
    r"\bquantum\s+tunnell?ing\b",
    r"\bquantum\s+computing\b",
    r"\bquantum\s+computer(?:s)?\b",
    r"\bquantum\s+gravity\b",
    r"\bquantum\s+optics\b",
    r"\bquantum\s+dot(?:s)?\b",
    r"\bquantum\s+well(?:s)?\b",
    r"\bquantum\s+spin\b",
    r"\bquantum\s+decoherence\b",
    r"\bquantum\s+information\b",
    r"\bquantum\s+cryptography\b",
    r"\bquantized\b",
    r"\bquantization\b",
    # NOTE: quanta of (energy|light|radiation|action) REMOVED (Planck 1900)
    r"\bwave\s*[\s-]function(?:s)?\b",           # 1926 Schrödinger
    r"\bwavefunction\b",
    r"\buncertainty\s+principle\b",              # 1927 Heisenberg
    r"\bexclusion\s+principle\b",                # 1925 Pauli
    r"\bpauli\s+exclusion\b",
    r"\bpauli\s+matrices\b",
    r"\bsuperposition\s+(?:of\s+)?(?:states?|wavefunctions?)\b",
    r"\bwave[- ]?particle\s+duality\b",
    r"\bde\s+broglie\b",                         # 1924
    r"\bcompton\s+(?:scattering|effect|wavelength)\b",  # 1923
    r"\bdirac\s+(?:equation|delta|notation|spinor|fermion|sea)\b",  # 1928
    r"\bfeynman\s+(?:diagram|path|integral|propagator|rule)\b",
    r"\bpath\s+integral(?:s)?\b",
    r"\brenormalization\b",
    r"\bcopenhagen\s+interpretation\b",
    r"\bmany[- ]worlds\s+interpretation\b",
    r"\bbell(?:'?s)?\s+(?:inequality|theorem)\b",  # 1964
    r"\bEPR\s+paradox\b",                        # 1935
    r"\bspin[- ]orbit\s+coupling\b",
    r"\bzero[- ]point\s+energy\b",

    # -- Relativity (all 1905+, still rejected) --
    r"\bspacetime\b",                            # 1908 Minkowski
    r"\bspace-time\b",
    r"\bspecial\s+relativity\b",                 # 1905 Einstein
    r"\bgeneral\s+relativity\b",                 # 1915 Einstein
    r"\brelativistic\b",
    r"\btime\s+dilation\b",
    r"\blength\s+contraction\b",
    r"\bmass[–-]energy\b",
    r"\be\s*=\s*mc\s*(?:\^|²|2)\b",
    r"\blorentz\s+(?:transformation|contraction|factor|invariance|covariance)\b",
    r"\bminkowski\s+(?:space|metric|diagram|geometry)\b",
    r"\bgravitational\s+(?:wave|lensing|redshift|time\s+dilation)\b",
    r"\bschwarzschild\s+(?:radius|metric|solution|singularity)\b",
    r"\bgeodesic(?:s)?\b(?=.*\b(?:spacetime|relativity|metric|tensor)\b)",
    r"\bframe[- ]dragging\b",
    r"\btwin\s+paradox\b",
    r"\bequivalence\s+principle\b(?=.*\b(?:einstein|gravity|inertial|relativity)\b)",

    # -- Einstein (all notable work is 1905+) --
    r"\beinstein(?:'?s)?\b",                     # reject any mention
    r"\bphotoelectric\s+effect\b",               # 1905 Einstein

    # -- Nuclear & particle physics --
    r"\bnuclear\s+(?:fission|fusion|reactor|bomb|weapon|physics|energy|force|decay|radiation|spin|magnetic|power|warhead|fallout|waste|fuel|chain)\b",
    r"\batomic\s+bomb(?:s)?\b",
    r"\batomic\s+energy\b",
    r"\batomic\s+weapon(?:s)?\b",
    r"\batomic\s+pile\b",                        # early name for reactor
    r"\batomic\s+number\b",                      # Moseley 1913
    r"\bchain\s+reaction\b(?=.*\b(?:nuclear|fission|uranium|plutonium|reactor)\b)",
    r"\buranium\s+(?:enrichment|fission|bomb|reactor|milling|hexafluoride|centrifuge)\b",
    r"\bplutonium\b",
    r"\btritium\b",                              # named 1934
    r"\bdeuterium\b",                            # named 1931
    r"\bcyclotron(?:s)?\b",                      # 1932
    r"\bsynchrotron(?:s)?\b",
    r"\bparticle\s+accelerator(?:s)?\b",
    r"\blinear\s+accelerator\b",
    r"\bcollider(?:s)?\b",
    # NOTE: radioactive decay REMOVED (1902 Rutherford/Soddy)
    # NOTE: radioactivity REMOVED (1898 Curie)
    r"\bradioactive\s+(?:isotope|material|waste|contamination|fallout|dating|tracer)\b",
    r"\bhalf[- ]life\b(?=.*\b(?:decay|radioactive|isotope|uranium|radium|thorium|carbon)\b)",
    r"\bisotope(?:s|ic)?\b",                     # 1913 Soddy
    r"\bstandard\s+model\b(?=.*\b(?:particle|physics|quark|lepton|boson|gauge|higgs)\b)",
    r"\bhiggs\s+(?:boson|field|particle|mechanism)\b",
    r"\bantimatter\b",
    r"\bquark(?:s)?\b",                          # 1964 Gell-Mann
    r"\blepton(?:s)?\b(?=.*\b(?:particle|electron|muon|neutrino|tau|physics|decay|flavor)\b)",
    r"\bboson(?:s)?\b(?=.*\b(?:particle|higgs|gauge|W\b|Z\b|photon|gluon|physics)\b)",
    r"\bgauge\s+(?:theory|invariance|symmetry|boson|field)\b",
    r"\bsymmetry\s+breaking\b",
    r"\bstrong\s+(?:force|interaction|nuclear)\b",
    r"\bweak\s+(?:force|interaction|nuclear)\b(?=.*\b(?:boson|decay|neutrino|parity|particle)\b)",
    r"\belectroweak\b",
    r"\bcolor\s+charge\b",

    # -- Cosmology --
    r"\bbig\s+bang(?:\s+theory)?\b",
    r"\bexpanding\s+universe\b",
    r"\bcosmic\s+(?:microwave|inflation|background|ray(?:s)?)\b",
    r"\bhubble(?:'?s)?\s+(?:constant|law|telescope|expansion)\b",
    r"\bredshift\b(?=.*\b(?:galaxy|galaxies|universe|cosmolog|hubble|doppler|recession)\b)",
    r"\bdark\s+matter\b(?=.*\b(?:galaxy|universe|cosmolog|particle|gravit|mass|halo|WIMP)\b)",
    r"\bdark\s+energy\b(?=.*\b(?:universe|cosmolog|expansion|accelerat|vacuum)\b)",
    r"\bcosmological\s+constant\b",              # Einstein 1917
    r"\bsteady[- ]state\s+(?:theory|universe|cosmology)\b",
    r"\bolbers(?:'?s)?\s+paradox\b",             # named 1952

    # -- Astrophysics post-1904 --
    r"\bevent\s+horizon(?:s)?\b",                # 1950s
    r"\bblack\s+hole(?:s)?\b(?=.*\b(?:singularity|event\s+horizon|hawking|gravity|mass|spacetime|relativity|stellar|supermassive|gravitational|accretion|ergosphere)\b)",
    r"\bneutron\s+star(?:s)?\b",
    r"\bpulsar(?:s)?\b",                         # 1967
    r"\bquasar(?:s)?\b",                         # 1963
    r"\bwhite\s+dwarf(?:s)?\b",
    r"\bred\s+giant(?:s)?\b(?=.*\b(?:star|stellar|evolution|main\s+sequence|supergiant)\b)",
    r"\bmain\s+sequence\b",                      # Hertzsprung 1911
    r"\bchandrasekhar\b",
    r"\bhawking\s+radiation\b",
    r"\bhertzsprung[- ]russell\b",               # 1911-1913
    r"\bstellar\s+(?:evolution|nucleosynthesis)\b",
    r"\bsupernova(?:e|s)?\b(?=.*\b(?:type\s+I|remnant|neutron|collapse|nucleosynthesis)\b)",

    # -- Condensed matter & other post-1904 --
    r"\bsuperconduct(?:ivity|ing|or(?:s)?)\b",   # 1911
    r"\bsuperfluid(?:ity)?\b",                   # 1937
    r"\bbose[- ]einstein\s+(?:condensat|statistic)\w*\b",
    r"\bfermi[- ]dirac\b",
    r"\bfermi\s+(?:level|energy|surface|gas|liquid)\b",
    r"\bband\s+gap\b",                           # solid state physics
    r"\bband\s+structure\b(?=.*\b(?:electron|solid|crystal|semiconductor|metal)\b)",
    r"\bphonon(?:s)?\b",
    r"\bplasmon(?:s)?\b",
    r"\bexciton(?:s)?\b",
    r"\bmagnon(?:s)?\b",

    # -- Post-1904 technology --
    r"\blaser(?:s)?\b",                          # 1960
    r"\btransistor(?:s)?\b",                     # 1947
    r"\bsemiconductor(?:s)?\b",
    r"\bintegrated\s+circuit(?:s)?\b",
    r"\bmicroprocessor(?:s)?\b",
    r"\bnuclear\s+(?:power\s+plant|submarine|aircraft\s+carrier)\b",
    r"\bgeiger\s+counter\b",                     # 1908
    r"\bcloud\s+chamber\b",                      # 1911 Wilson
    r"\bbubble\s+chamber\b",                     # 1952
    r"\belectron\s+microscop(?:e|y)\b",          # 1931
    r"\bmass\s+spectromet(?:er|ry)\b",           # 1918
    r"\bnuclear\s+magnetic\s+resonance\b",       # 1938
    r"\b(?:NMR|MRI)\b",
    r"\bradar\b",                                # 1930s
    r"\bsonar\b",                                # 1906
    r"\batomic\s+clock\b",
    r"\bcarbon[- ](?:14|fourteen)\s+dating\b",

    # -- Named physicists (unambiguous post-1904 work) --
    r"\bschr[öo]dinger(?:'?s)?\b",
    r"\bheisenberg(?:'?s)?\b",
    r"\boppenheimer(?:'?s)?\b",                  # Manhattan Project
    r"\bfeynman(?:'?s)?\b",
    r"\bbohr(?:'?s)?\s+(?:model|radius|atom|magneton|theory|postulate)\b",  # 1913
    # NOTE: planck constant/law/... REMOVED (Planck 1900)
    r"\brutherford(?:'?s)?\s+(?:model|experiment|scattering|atom|gold\s+foil)\b",  # 1911
    r"\bdirac(?:'?s)?\b(?=.*\b(?:equation|delta|fermion|sea|spinor|notation|bracket|hole)\b)",
    r"\bpauli(?:'?s)?\b(?=.*\b(?:exclusion|principle|matrices|spin|equation)\b)",
    r"\bfermi(?:'?s)?\b(?=.*\b(?:level|energy|surface|gas|statistics|interaction|paradox|golden)\b)",
    r"\bborn(?:'?s)?\s+(?:approximation|rule|interpretation)\b",
    r"\bsommerfeld\b(?=.*\b(?:model|expansion|constant|quantum)\b)",
    r"\bgamow\b",                                # 1928+ (tunneling, Big Bang)
    r"\blema[iî]tre\b",                          # 1927 expanding universe
    r"\bhubble\b(?=.*\b(?:telescope|constant|law|expansion|redshift)\b)",

    # -- Proton/neutron as physics particles --
    r"\bproton(?:s)?\b(?=.*\b(?:electron|neutron|nucleus|particle|charge|mass|accelerat|collid|beam|decay|spin|antiproton)\b)",
    r"\bneutron(?:s)?\b(?=.*\b(?:proton|electron|nucleus|particle|fission|reactor|bomb|star|decay|mass|capture|scattering)\b)",

    # -- Electron in post-1904 physics contexts --
    r"\belectron\s+(?:spin|orbital|shell|cloud|beam|microscop|volt|capture|configuration|diffraction|density|affinity|gun|pair)\b",
    r"\belectron[- ]positron\b",
    r"\bfree\s+electron\b(?=.*\b(?:laser|gas|metal|Fermi|band|Drude)\b)",

    # -- String theory & modern --
    r"\bstring\s+theory\b",
    r"\bsupersymmetry\b",
    r"\bsuperstring\b",
    r"\bM[- ]theory\b",
    r"\bextra\s+dimensions?\b(?=.*\b(?:string|compact|Kaluza|Klein|brane)\b)",
    r"\bbrane\b(?=.*\b(?:string|dimension|bulk|theory)\b)",
    r"\bloop\s+quantum\s+gravity\b",

    # -- Post-1904 institutions/projects --
    r"\bmanhattan\s+project\b",
    r"\blos\s+alamos\b(?=.*\b(?:nuclear|bomb|laboratory|weapon|test)\b)",
    r"\boak\s+ridge\b(?=.*\b(?:nuclear|uranium|reactor|laboratory|national)\b)",
    r"\b(?:CERN|Fermilab|SLAC|DESY|KEK|Brookhaven)\b",
    r"\blarge\s+hadron\s+collider\b",
    r"\batomic\s+energy\s+commission\b",
]

# Context patterns: these terms are common in pre-1904 text with different
# meanings. Only reject if 3+ co-occur in the same document, which signals
# a physics-heavy modern text that slipped through.
CONTEXT_PATTERNS = [
    r"\bquantum\b",             # legal: "quantum meruit", "quantum of damages"
    r"\bnuclear\b",             # biology: "nuclear membrane", "nuclear division"
    r"\bblack\s+hole(?:s)?\b",  # literal: prison, dark place, Calcutta
    r"\bdark\s+matter\b",       # literary: "dark matter here below"
    r"\bdark\s+energy\b",       # literary: "dark energy of whose course"
    r"\bfusion\b",              # general: merger, joining
    r"\bfission\b",             # biology: cell fission
    r"\bradiation\b",           # general: heat radiation (pre-1900 OK)
    r"\bspectrum\b",            # general: light spectrum (pre-1900 OK)
    r"\belectron(?:s)?\b",      # existed from 1894 but rare pre-1900
]

CONTEXT_THRESHOLD = 3

CHARS_PER_TOKEN = 4.2


def build_year_regex(end_year: int) -> re.Pattern:
    """Build a regex that matches years > end_year (up to 2029).

    For end_year=1904: matches 1905-2029
    For end_year=1909: matches 1910-2029
    """
    post_year = end_year + 1
    parts = []

    # Handle individual years in the same decade as post_year
    decade_start = (post_year // 10) * 10
    decade_end = decade_start + 9
    century = post_year // 100  # e.g. 19

    if post_year <= decade_end:
        # Match post_year through end of its decade
        first_digit = post_year % 10
        if first_digit == 0:
            # Whole decade: e.g. 1910-1919 -> 191\d
            parts.append(f"{decade_start // 10}\\d")
        elif first_digit == 9:
            # Single year: e.g. 1909
            parts.append(str(post_year))
        else:
            # Range within decade: e.g. 1905-1909 -> 190[5-9]
            parts.append(f"{decade_start // 10}[{first_digit}-9]")

    # Handle remaining full decades in the same century
    next_decade = decade_start + 10
    century_end_decade = century * 100 + 90
    if next_decade <= century_end_decade:
        next_decade_digit = next_decade // 10 % 10
        century_end_digit = century_end_decade // 10 % 10
        if next_decade_digit == century_end_digit:
            parts.append(f"{century}{next_decade_digit}\\d")
        else:
            parts.append(f"{century}[{next_decade_digit}-{century_end_digit}]\\d")

    # Handle 20xx years (2000-2029)
    if century < 20:
        parts.append(r"20[0-2]\d")

    pattern = r"\b(?:" + "|".join(parts) + r")\b"
    return re.compile(pattern)


# Default: catches 1905+ for end_year=1904
POST_CUTOFF_YEAR_RE = build_year_regex(1904)
POST_CUTOFF_YEAR_THRESHOLD = 5


# ============================================================================
# Document Chunking
# ============================================================================
CHUNK_SENTENCE_RE = re.compile(r'[.!?]\s')


def chunk_document(text: str, max_chars: int = 8000, min_chars: int = 200) -> list[str]:
    """Split a long document into chunks of roughly max_chars at natural boundaries."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_pos = window.rfind('\n\n')

        if split_pos > max_chars // 4:
            split_pos += 2
        else:
            match = None
            for m in CHUNK_SENTENCE_RE.finditer(window):
                match = m
            if match and match.end() > max_chars // 4:
                split_pos = match.end()
            else:
                split_pos = max_chars

        chunk = remaining[:split_pos].rstrip()
        if len(chunk) >= min_chars:
            chunks.append(chunk)
        remaining = remaining[split_pos:].lstrip()

    if len(remaining.strip()) >= min_chars:
        chunks.append(remaining.strip())

    return chunks


# ============================================================================
# Dask worker function — filters supplement shards with year + anachronism
# ============================================================================

def filter_shard_batch(
    batch_json: str,
    end_year: int,
    year_regex_pattern: str,
) -> dict:
    """
    Process a batch of (input_path, output_path) tuples from the supplement.

    For each shard: read row-group by row-group, filter by year column
    (keep only year <= end_year), then apply anachronism filter to text.
    Write clean docs to staging_dir.

    Returns dict with stats.
    """
    import json as _json
    import gc as _gc
    import os as _os
    import re as _re
    import pyarrow as _pa
    import pyarrow.parquet as _pq

    # Compile patterns inside worker (can't serialize compiled regex across Dask)
    always_reject = _re.compile(
        '|'.join(ALWAYS_REJECT_PATTERNS), _re.IGNORECASE
    )
    context_pats = [_re.compile(p, _re.IGNORECASE) for p in CONTEXT_PATTERNS]
    year_pat = _re.compile(year_regex_pattern)

    items = _json.loads(batch_json)
    total_in = 0
    total_kept = 0
    total_rejected = 0
    total_chars = 0
    reasons: dict[str, int] = {}

    for input_path, output_path in items:
        try:
            pf = _pq.ParquetFile(input_path)
        except Exception:
            continue

        # Check if this file has a year column
        has_year_col = "year" in pf.schema.names

        clean_texts: list[str] = []

        for rg_idx in range(pf.num_row_groups):
            if has_year_col:
                table = pf.read_row_group(rg_idx, columns=["text", "year"])
                texts = table.column("text").to_pylist()
                years = table.column("year").to_pylist()
            else:
                table = pf.read_row_group(rg_idx, columns=["text"])
                texts = table.column("text").to_pylist()
                years = [None] * len(texts)
            del table

            for text, year in zip(texts, years):
                total_in += 1

                # Year filter: reject docs with year > end_year
                if year is not None and year > end_year:
                    total_rejected += 1
                    reasons["year_too_new"] = reasons.get("year_too_new", 0) + 1
                    continue

                if not text or len(text) < 500:
                    total_rejected += 1
                    reasons["too_short"] = reasons.get("too_short", 0) + 1
                    continue

                m = always_reject.search(text)
                if m:
                    total_rejected += 1
                    bucket = m.group()
                    if len(bucket) > 40:
                        bucket = bucket[:40]
                    reasons[bucket] = reasons.get(bucket, 0) + 1
                    continue

                year_matches = year_pat.findall(text)
                if len(year_matches) >= POST_CUTOFF_YEAR_THRESHOLD:
                    total_rejected += 1
                    reasons["post_cutoff_dates"] = reasons.get("post_cutoff_dates", 0) + 1
                    continue

                context_hits = 0
                rejected = False
                for pat in context_pats:
                    if pat.search(text):
                        context_hits += 1
                        if context_hits >= CONTEXT_THRESHOLD:
                            total_rejected += 1
                            reasons["context_accumulation"] = reasons.get("context_accumulation", 0) + 1
                            rejected = True
                            break
                if rejected:
                    continue

                clean_texts.append(text)
                total_kept += 1
                total_chars += len(text)

        # Write clean docs for this shard
        if clean_texts:
            _os.makedirs(_os.path.dirname(output_path), exist_ok=True)
            out_table = _pa.table({"text": clean_texts})
            _pq.write_table(out_table, output_path, compression="zstd")
            del out_table
        del clean_texts
        _gc.collect()

    return {
        "n_in": total_in,
        "n_kept": total_kept,
        "n_rejected": total_rejected,
        "n_chars": total_chars,
        "reasons": reasons,
    }


# ============================================================================
# Adaptive batching by file size
# ============================================================================

def make_adaptive_batches(pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group (input_path, output_path) pairs into batches based on file size."""
    sized = []
    for inp, out in pairs:
        try:
            sz = os.path.getsize(inp)
        except OSError:
            sz = 0
        sized.append((inp, out, sz))

    sized.sort(key=lambda x: -x[2])

    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    current_batch_size = 0

    for inp, out, sz in sized:
        if sz > 500_000_000:
            max_per_batch = 1
        elif sz > 100_000_000:
            max_per_batch = 4
        elif sz > 10_000_000:
            max_per_batch = 16
        else:
            max_per_batch = 64

        if current_batch and (len(current_batch) >= max_per_batch or current_batch_size > 500_000_000):
            batches.append(current_batch)
            current_batch = []
            current_batch_size = 0

        current_batch.append((inp, out))
        current_batch_size += sz

        if len(current_batch) >= max_per_batch:
            batches.append(current_batch)
            current_batch = []
            current_batch_size = 0

    if current_batch:
        batches.append(current_batch)

    return batches


# ============================================================================
# Main
# ============================================================================

def main():
    import json
    from evolutionaryscale.utils.executor import DaskExecutorContext, TaskGranularity  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(
        description="Filter supplement by year and anachronism, combine with pre-1900 data, reshard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--end-year", type=int, default=1904,
                        help="Include documents up to and including this year (e.g. 1904 for a pre-1905 split)")
    parser.add_argument("--pre1900-staging", type=Path,
                        default=Path("/mnt/main0/data/michaelhla/pre1900_full_clean/_staging"),
                        help="Pre-1900 staged data directory (already anachronism-filtered, text only)")
    parser.add_argument("--supplement-raw", type=Path,
                        default=Path("/mnt/main0/data/michaelhla/pre1915_supplement_raw"),
                        help="1900-1914 supplement raw data directory (has year column)")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path("/mnt/main0/data/michaelhla/pre1905_full_clean"),
                        help="Output directory for combined clean parquet shards")
    parser.add_argument("--phase", choices=["both", "filter", "reshard"], default="both",
                        help="Which phase to run (filter=Phase 1, reshard=Phase 2, both=both)")
    parser.add_argument("--docs-per-shard", type=int, default=65536,
                        help="Target documents per output shard (1024 * 1 RG/GPU * 64 GPUs)")
    parser.add_argument("--row-group-size", type=int, default=1024,
                        help="Row group size within each parquet file (matches dataloader batch granularity)")
    parser.add_argument("--val-fraction", type=float, default=0.005,
                        help="Fraction of data for validation (last shard)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Max characters per document chunk")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Min characters per chunk")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of Dask workers")
    parser.add_argument("--mem-per-worker", type=int, default=48,
                        help="Memory per worker in GB")
    parser.add_argument("--partition", type=str, default="midpri",
                        help="Slurm partition")
    parser.add_argument("--world-size", type=int, default=64,
                        help="DDP world size (num GPUs). Row groups per shard will be divisible by this.")
    parser.add_argument("--clean", action="store_true",
                        help="Clear staging and output directories before starting")
    args = parser.parse_args()

    import shutil
    random.seed(args.seed)

    # Build year regex for the specified cutoff
    year_re = build_year_regex(args.end_year)
    year_regex_pattern = year_re.pattern

    if args.clean:
        print("Cleaning output and staging directories...")
        if args.output.exists():
            shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)
    staging_dir = args.output / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    print(f"Year split: end_year={args.end_year} (keeping documents <= {args.end_year})")
    print(f"Date regex: catches {args.end_year + 1}+ -> {year_regex_pattern}")

    # ================================================================
    # Phase 1: Dask-parallel filtering of SUPPLEMENT (year + anachronism)
    # ================================================================
    if args.phase in ("both", "filter"):
        print(f"\n{'=' * 60}")
        print(f"PHASE 1: FILTERING SUPPLEMENT (year <= {args.end_year} + anachronism)")
        print(f"{'=' * 60}")
        print(f"  Supplement input: {args.supplement_raw}")
        print(f"  Pre-1900 staged data: {args.pre1900_staging} (passed through in Phase 2)")

        # Discover all parquet files in supplement
        all_files: list[tuple[Path, int]] = []
        for f in sorted(args.supplement_raw.rglob("*.parquet")):
            try:
                sz = os.path.getsize(f)
                all_files.append((f, sz))
            except Exception:
                pass

        total_size = sum(sz for _, sz in all_files)
        print(f"\n  Supplement files: {len(all_files)}, {total_size / 1e9:.1f} GB compressed")

        # Build (input, output) pairs, skipping already-staged files
        pairs: list[tuple[str, str]] = []
        n_already = 0
        for fpath, _ in all_files:
            # Use relative path components as prefix to avoid collisions
            rel = fpath.relative_to(args.supplement_raw)
            out_name = f"supplement_{rel.parent.name}_{fpath.stem}.parquet"
            out_path = str(staging_dir / out_name)
            if os.path.exists(out_path):
                n_already += 1
            else:
                pairs.append((str(fpath), out_path))

        if n_already > 0:
            print(f"\n  Resuming: {n_already} files already filtered, {len(pairs)} remaining")

        if not pairs:
            print("\n  All supplement files already filtered. Skipping to Phase 2.")
        else:
            batches = make_adaptive_batches(pairs)
            random.shuffle(batches)

            print(f"\n  Filtering {len(pairs)} supplement files in {len(batches)} adaptive batches")
            print(f"  Workers: {args.workers}, Mem/worker: {args.mem_per_worker} GB")

            t0 = time.time()

            batch_jsons = [json.dumps(batch) for batch in batches]

            with DaskExecutorContext(
                task_granularity=TaskGranularity.CPU,
                slurm_partition=args.partition,
                num_jobs=min(args.workers, len(batches)),
                cpus_per_worker=1,
                mem_per_worker_gb=args.mem_per_worker,
            ) as executor:
                results = executor.map(
                    filter_shard_batch,
                    batch_jsons,
                    [args.end_year] * len(batches),
                    [year_regex_pattern] * len(batches),
                    progress=True,
                    errors="skip",
                )

            # Aggregate stats
            total_in = 0
            total_kept = 0
            total_rejected = 0
            total_chars = 0
            n_failed = 0
            all_reasons: dict[str, int] = {}

            for result in results:
                if result is None:
                    n_failed += 1
                    continue
                total_in += result["n_in"]
                total_kept += result["n_kept"]
                total_rejected += result["n_rejected"]
                total_chars += result["n_chars"]
                for r, c in result["reasons"].items():
                    all_reasons[r] = all_reasons.get(r, 0) + c

            elapsed = time.time() - t0

            print(f"\n{'=' * 60}")
            print(f"PHASE 1 COMPLETE (supplement filter, year <= {args.end_year})")
            print(f"{'=' * 60}")
            print(f"  Input docs:    {total_in:,}")
            print(f"  Kept:          {total_kept:,} ({100 * total_kept / max(total_in, 1):.1f}%)")
            print(f"  Rejected:      {total_rejected:,} ({100 * total_rejected / max(total_in, 1):.1f}%)")
            print(f"  Clean chars:   {total_chars / 1e9:.2f}B (~{total_chars / CHARS_PER_TOKEN / 1e9:.2f}B tokens)")
            if n_failed:
                print(f"  Failed batches: {n_failed}")
            print(f"  Elapsed:       {elapsed:.0f}s ({elapsed / 3600:.2f}h)")

            if all_reasons:
                print(f"\n  Top rejection reasons:")
                for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1])[:20]:
                    print(f"    {reason}: {count:,}")

    # ================================================================
    # Phase 2: Combine pre-1900 + filtered supplement, chunk, reshard
    # ================================================================
    if args.phase in ("both", "reshard"):
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: STREAMING CHUNK AND RESHARD (combined corpus)")
        print(f"{'=' * 60}")
        print(f"  Chunking: max_chars={args.max_chars}, min_chars={args.min_chars}")
        t1 = time.time()

        # Collect input files from TWO sources:
        # 1. Pre-1900 staged data (already anachronism-filtered, text only)
        pre1900_files = sorted(args.pre1900_staging.glob("*.parquet"))
        # 2. Filtered supplement (from staging dir)
        supplement_files = sorted(staging_dir.glob("*.parquet"))

        all_input_files = list(pre1900_files) + list(supplement_files)
        random.shuffle(all_input_files)

        print(f"  Pre-1900 files: {len(pre1900_files)}")
        print(f"  Supplement files (filtered): {len(supplement_files)}")
        print(f"  Total input files: {len(all_input_files)}")

        if not all_input_files:
            print("  ERROR: No input files found. Run --phase filter first.")
            return

        # Compute docs_per_shard so that num_row_groups is divisible by world_size
        rgs_per_gpu = max(1, round(args.docs_per_shard / (args.row_group_size * args.world_size)))
        docs_per_shard = args.row_group_size * rgs_per_gpu * args.world_size
        n_rg_per_shard = docs_per_shard // args.row_group_size

        print(f"  Target: {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups/shard "
              f"({rgs_per_gpu} per GPU x {args.world_size} GPUs)")

        # Streaming one-pass
        BUFFER_SHARDS = 10
        buffer_capacity = BUFFER_SHARDS * docs_per_shard

        buffer: list[str] = []
        val_reservoir: list[str] = []
        total_chunks_seen = 0
        total_chars = 0
        total_raw_docs = 0
        total_chunked_docs = 0
        shard_idx = 0
        shard_char_counts: list[int] = []

        def flush_buffer(buf: list[str], start_shard: int) -> int:
            """Sort + round-robin distribute buffer into shards, write them."""
            if not buf:
                return 0
            n_shards = len(buf) // docs_per_shard
            if n_shards == 0:
                return 0
            n_use = n_shards * docs_per_shard
            to_shard = buf[:n_use]

            to_shard.sort(key=len)
            buckets: list[list[str]] = [[] for _ in range(n_shards)]
            for i, chunk in enumerate(to_shard):
                buckets[i % n_shards].append(chunk)

            for j, bucket in enumerate(buckets):
                random.shuffle(bucket)
                out_path = args.output / f"shard_{start_shard + j:05d}.parquet"
                table = pa.table({"text": bucket})
                pq.write_table(table, out_path,
                               row_group_size=args.row_group_size,
                               compression="zstd",
                               write_statistics=False,
                               use_dictionary=False)
                shard_char_counts.append(sum(len(c) for c in bucket))
                del table

            written = n_shards
            if (start_shard + written) % 50 < written or start_shard == 0:
                print(f"    Written shards {start_shard:,}-{start_shard + written - 1:,}")

            del buf[:n_use]
            gc.collect()
            return written

        print(f"\n  Streaming through all input files (buffer={BUFFER_SHARDS} shards)...")

        for file_idx, sf in enumerate(all_input_files):
            try:
                pf = pq.ParquetFile(sf)
                for rg_idx in range(pf.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=["text"])
                    texts = table.column("text").to_pylist()
                    del table
                    for text in texts:
                        if not text:
                            continue
                        total_raw_docs += 1
                        doc_chunks = chunk_document(text, max_chars=args.max_chars, min_chars=args.min_chars)
                        if len(doc_chunks) > 1:
                            total_chunked_docs += 1
                        for c in doc_chunks:
                            total_chunks_seen += 1
                            total_chars += len(c)
                            if random.random() < args.val_fraction:
                                val_reservoir.append(c)
                            else:
                                buffer.append(c)
            except Exception:
                continue

            if len(buffer) >= buffer_capacity:
                n_written = flush_buffer(buffer, shard_idx)
                shard_idx += n_written

            if (file_idx + 1) % 500 == 0:
                est_tok = total_chars / CHARS_PER_TOKEN
                print(f"    Processed {file_idx + 1}/{len(all_input_files)} files, "
                      f"{total_chunks_seen:,} chunks, ~{est_tok / 1e9:.2f}B tokens, "
                      f"{shard_idx} shards written")

        # Final flush of remaining buffer
        if len(buffer) >= docs_per_shard:
            n_written = flush_buffer(buffer, shard_idx)
            shard_idx += n_written

        # Any leftover (< docs_per_shard) goes into val
        val_reservoir.extend(buffer)
        buffer.clear()
        gc.collect()

        n_train_shards = shard_idx

        # Write val shard (always last, as expected by dataloader)
        print(f"  Writing validation shard (shard_{shard_idx:05d}), {len(val_reservoir):,} chunks...")
        val_path = args.output / f"shard_{shard_idx:05d}.parquet"
        random.shuffle(val_reservoir)
        table = pa.table({"text": val_reservoir})
        pq.write_table(table, val_path,
                        row_group_size=args.row_group_size,
                        compression="zstd",
                        write_statistics=False,
                        use_dictionary=False)
        del table
        shard_idx += 1
        del val_reservoir
        gc.collect()

        elapsed2 = time.time() - t1
        est_tokens = total_chars / CHARS_PER_TOKEN

        # Report token balance
        if shard_char_counts:
            avg_c = sum(shard_char_counts) / len(shard_char_counts)
            min_c = min(shard_char_counts)
            max_c = max(shard_char_counts)
            imbalance = (max_c - min_c) / avg_c * 100
            print(f"\n  Token balance: avg={avg_c / CHARS_PER_TOKEN / 1e6:.1f}M tok/shard, "
                  f"min={min_c / CHARS_PER_TOKEN / 1e6:.1f}M, "
                  f"max={max_c / CHARS_PER_TOKEN / 1e6:.1f}M, "
                  f"imbalance={imbalance:.2f}%")

        # Report file sizes
        shard_sizes = []
        for i in range(n_train_shards):
            sp = args.output / f"shard_{i:05d}.parquet"
            if sp.exists():
                shard_sizes.append(os.path.getsize(sp))

        print(f"\n{'=' * 60}")
        print(f"DONE")
        print(f"{'=' * 60}")
        print(f"  Docs: {total_raw_docs:,} raw -> {total_chunks_seen:,} chunks ({total_chunked_docs:,} split)")
        print(f"  Train: {n_train_shards} shards")
        print(f"    {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups (÷ {args.world_size} = {rgs_per_gpu} per GPU)")
        if shard_sizes:
            print(f"    File sizes: avg={sum(shard_sizes) / len(shard_sizes) / 1e6:.1f}MB, "
                  f"min={min(shard_sizes) / 1e6:.1f}MB, max={max(shard_sizes) / 1e6:.1f}MB")
        print(f"  Val:   1 shard (shard_{n_train_shards:05d})")
        print(f"  Total: {shard_idx} shards")
        print(f"  Est. tokens: {est_tokens / 1e9:.2f}B")
        print(f"  Output: {args.output}")
        print(f"  Phase 2 elapsed: {elapsed2:.0f}s")


if __name__ == "__main__":
    main()
