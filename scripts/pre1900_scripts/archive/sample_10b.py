#!/usr/bin/env python3
"""
Sample ~10B tokens from the filtered pre-1900 corpus, with comprehensive
post-1900 physics filtering applied to EVERY document's FULL text.
Parallelized with Dask.

Two phases:
  Phase 1 (Dask-parallel): Each worker filters a batch of input shards and
      writes per-shard cleaned parquet files to a staging directory.
  Phase 2 (serial): Reads the staged files, shuffles, and writes final output
      shards in the format expected by the nanochat data loader.

Output format:
    shard_XXXXX.parquet  (text column only, snappy compression)
    Last shard = validation split

Usage:
    python scripts/pre1900_scripts/sample_10b.py \
        --input /mnt/main0/data/michaelhla/pre1900_filtered \
        --output /mnt/main0/data/michaelhla/pre1900_10b \
        --target-tokens 10e9 \
        --workers 64
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
# Comprehensive Post-1900 Physics Filter
# ============================================================================
# Everything below was coined, discovered, or formalized after 1900.
# We check the FULL text of every document (not just a sample).
#
# Three layers of defense:
#   1. ALWAYS_REJECT: unambiguous post-1900 physics/science terms — single match rejects.
#   2. DATE_REJECT: documents with many explicit post-1900 year references — catches
#      mislabeled government reports, court cases, statistical abstracts.
#   3. CONTEXT_REJECT: terms with innocent pre-1900 meanings — require 3+ co-occurrences.
# ============================================================================

ALWAYS_REJECT_PATTERNS = [
    # -- Particles & quantum objects (no pre-1900 meaning) --
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

    # -- Quantum mechanics terminology --
    r"\bquantum\s+mechan",                       # quantum mechanics/mechanical
    r"\bquantum\s+field",                        # quantum field theory
    r"\bquantum\s+electrodynamics\b",
    r"\bquantum\s+chromodynamics\b",
    r"\bquantum\s+theory\b",
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
    r"\bquanta\s+of\s+(?:energy|light|radiation|action)\b",
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

    # -- Relativity --
    r"\bspacetime\b",                            # 1908 Minkowski
    r"\bspace-time\b",
    r"\bspecial\s+relativity\b",
    r"\bgeneral\s+relativity\b",
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

    # -- Einstein (born 1879, but all notable work is post-1900) --
    r"\beinstein(?:'?s)?\b",                      # reject any mention
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
    r"\bradioactive\s+decay\b",
    r"\bradioactivity\b",                        # 1898 Curie
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
    r"\bolbers(?:'?s)?\s+paradox\b",             # named 1952 (though concept older)

    # -- Astrophysics post-1900 --
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
    r"\bhertzsprung[- ]russell\b",
    r"\bstellar\s+(?:evolution|nucleosynthesis)\b",
    r"\bsupernova(?:e|s)?\b(?=.*\b(?:type\s+I|remnant|neutron|collapse|nucleosynthesis)\b)",

    # -- Condensed matter & other post-1900 --
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

    # -- Post-1900 technology --
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

    # -- Named physicists (unambiguous — these names have no common pre-1900 usage) --
    r"\bschr[öo]dinger(?:'?s)?\b",
    r"\bheisenberg(?:'?s)?\b",
    r"\boppenheimer(?:'?s)?\b",                  # Manhattan Project
    r"\bfeynman(?:'?s)?\b",
    r"\bbohr(?:'?s)?\s+(?:model|radius|atom|magneton|theory|postulate)\b",
    r"\bplanck(?:'?s)?\s+(?:constant|law|length|time|mass|scale|epoch|radiation|formula)\b",
    r"\brutherford(?:'?s)?\s+(?:model|experiment|scattering|atom|gold\s+foil)\b",
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

    # -- Electron in post-1900 physics contexts --
    # "electron" itself existed from 1894 (Stoney) and Thomson 1897, but
    # most detailed discussion (orbitals, shells, spin, beams) is post-1900
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

    # -- Post-1900 institutions/projects (catches mislabeled docs) --
    r"\bmanhattan\s+project\b",
    r"\blos\s+alamos\b(?=.*\b(?:nuclear|bomb|laboratory|weapon|test)\b)",
    r"\boak\s+ridge\b(?=.*\b(?:nuclear|uranium|reactor|laboratory|national)\b)",
    r"\b(?:CERN|Fermilab|SLAC|DESY|KEK|Brookhaven)\b",
    r"\blarge\s+hadron\s+collider\b",
    r"\batomic\s+energy\s+commission\b",
]

# Context patterns: these terms are common in pre-1900 text with different
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

# Date-based rejection: if a document has many explicit references to years
# after 1900, it's almost certainly a mislabeled post-1900 document.
# This catches government reports, court cases, statistical abstracts, etc.
# that slipped through with wrong year metadata.
POST_1900_YEAR_RE = re.compile(
    r"\b(?:19\d{2}|20[0-2]\d)\b"  # matches 1900-2029
)
# Minimum count of post-1900 year mentions to trigger rejection.
# A genuine pre-1900 text might mention "1900" once or twice in a
# forward-looking context, but 5+ is a strong signal.
POST_1900_YEAR_THRESHOLD = 5

CHARS_PER_TOKEN = 4.2

# ============================================================================
# Document Chunking
# ============================================================================
# Long documents waste most of their tokens in BOS-aligned dataloaders.
# Splitting at ~8K chars (~2000 tokens) gives ~10x more effective tokens.

CHUNK_SENTENCE_RE = re.compile(r'[.!?]\s')


def chunk_document(text: str, max_chars: int = 8000, min_chars: int = 200) -> list[str]:
    """Split a long document into chunks of roughly max_chars at natural boundaries.

    Priority: split at paragraph breaks (\\n\\n), then sentence boundaries (. ! ?),
    then hard-split as a last resort. Chunks shorter than min_chars are dropped.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    remaining = text

    while len(remaining) > max_chars:
        window = remaining[:max_chars]
        split_pos = window.rfind('\n\n')

        if split_pos > max_chars // 4:
            split_pos += 2  # include the \n\n with the first chunk
        else:
            # Try sentence boundary
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
# Dask worker function
# ============================================================================

def filter_shard_batch(
    batch_json: str,
    staging_dir: str,
) -> dict:
    """
    Process a batch of (input_path, output_path) tuples.

    For each shard: read row-group by row-group, apply post-1900 physics
    filter to FULL text, write clean docs to staging_dir.

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
    year_pat = _re.compile(r"\b(?:19\d{2}|20[0-2]\d)\b")

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

        clean_texts: list[str] = []

        for rg_idx in range(pf.num_row_groups):
            table = pf.read_row_group(rg_idx, columns=["text"])
            texts = table.column("text").to_pylist()
            del table

            for text in texts:
                total_in += 1

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
                if len(year_matches) >= POST_1900_YEAR_THRESHOLD:
                    total_rejected += 1
                    reasons["post_1900_dates"] = reasons.get("post_1900_dates", 0) + 1
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
            _pq.write_table(out_table, output_path, compression="snappy")
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
# Main
# ============================================================================

def main():
    import json
    from evolutionaryscale.utils.executor import DaskExecutorContext, TaskGranularity  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser(
        description="Sample ~10B tokens from filtered pre-1900 corpus with thorough post-1900 physics removal (Dask-parallelized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_filtered"),
                        help="Filtered data directory (with source subdirectories)")
    parser.add_argument("--output", "-o", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_10b"),
                        help="Output directory for clean parquet shards")
    parser.add_argument("--target-tokens", type=float, default=10e9,
                        help="Target number of tokens (approximate)")
    parser.add_argument("--docs-per-shard", type=int, default=48000,
                        help="Target documents per output shard (rounded to make row groups divisible by world-size)")
    parser.add_argument("--row-group-size", type=int, default=500,
                        help="Row group size within each parquet file")
    parser.add_argument("--val-fraction", type=float, default=0.005,
                        help="Fraction of data for validation (last shard)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--newspaper-fraction", type=float, default=0.30,
                        help="Target fraction of tokens from newspapers (rest from books)")
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Max characters per document chunk (long docs split at natural boundaries)")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Min characters per chunk (shorter chunks dropped)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of Dask workers")
    parser.add_argument("--shards-per-batch", type=int, default=32,
                        help="Number of input shards per Dask worker batch")
    parser.add_argument("--mem-per-worker", type=int, default=48,
                        help="Memory per worker in GB (needs ~20-30 GB for large book shards)")
    parser.add_argument("--partition", type=str, default="midpri",
                        help="Slurm partition")
    parser.add_argument("--world-size", type=int, default=8,
                        help="DDP world size (num GPUs). Row groups per shard will be divisible by this.")
    parser.add_argument("--clean", action="store_true",
                        help="Clear staging and output directories before starting")
    args = parser.parse_args()

    import shutil
    random.seed(args.seed)

    if args.clean:
        print("Cleaning output and staging directories...")
        if args.output.exists():
            shutil.rmtree(args.output)

    args.output.mkdir(parents=True, exist_ok=True)
    staging_dir = args.output / "_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    target_chars = args.target_tokens * CHARS_PER_TOKEN
    target_news_chars = target_chars * args.newspaper_fraction
    target_book_chars = target_chars * (1 - args.newspaper_fraction)

    print(f"Target: {args.target_tokens/1e9:.0f}B tokens (~{target_chars/1e9:.0f}B chars)")
    print(f"  Books:      {target_book_chars/1e9:.1f}B chars ({1 - args.newspaper_fraction:.0%})")
    print(f"  Newspapers: {target_news_chars/1e9:.1f}B chars ({args.newspaper_fraction:.0%})")
    print()

    # ----------------------------------------------------------------
    # 1. Discover sources and estimate how many shards we need
    # ----------------------------------------------------------------
    print("Discovering sources...")
    # Use institutional_split (pre-split small files) instead of institutional
    # (multi-GB single-row-group files that require huge workers).
    # Pre-split with: scripts/pre1900_scripts/presplit_large.py
    book_sources = ["blbooks", "books", "institutional_split"]
    news_sources = ["newspapers"]

    def get_shards_with_sizes(subdirs):
        shards = []
        for subdir_name in subdirs:
            subdir = args.input / subdir_name
            if not subdir.is_dir():
                continue
            for s in sorted(subdir.glob("*.parquet")):
                try:
                    sz = os.path.getsize(s)
                    shards.append((s, sz))
                except Exception:
                    pass
        return shards

    book_shards_with_sizes = get_shards_with_sizes(book_sources)
    news_shards_with_sizes = get_shards_with_sizes(news_sources)

    print(f"  Book shards available: {len(book_shards_with_sizes)}")
    print(f"  News shards available: {len(news_shards_with_sizes)}")

    # Shuffle and select enough shards to (over-)cover target.
    # Parquet text is typically ~2-3x the file size when decompressed,
    # and we filter some docs out, so grab ~1.5x what we need by file size.
    random.shuffle(book_shards_with_sizes)
    random.shuffle(news_shards_with_sizes)

    def select_shards(shards_with_sizes, target_chars, label):
        """Select enough shards to likely cover target_chars.
        Compressed parquet is ~2-3x smaller than raw text."""
        selected = []
        cum_size = 0
        # Rough heuristic: need ~target_chars bytes of compressed parquet
        # (parquet text compresses ~2-3x, but we also filter ~5-15% of docs)
        target_file_bytes = target_chars * 2.0  # book parquets are ~0.4 chars/byte compressed
        for shard_path, sz in shards_with_sizes:
            selected.append(shard_path)
            cum_size += sz
            if cum_size >= target_file_bytes:
                break
        print(f"  Selected {len(selected)} {label} shards ({cum_size/1e9:.1f} GB)")
        return selected

    book_shards = select_shards(book_shards_with_sizes, target_book_chars, "book")
    news_shards = select_shards(news_shards_with_sizes, target_news_chars, "news")

    # ----------------------------------------------------------------
    # 2. Phase 1: Parallel filtering with Dask
    # ----------------------------------------------------------------
    def build_pairs(shards, label):
        pairs = []
        for shard_path in shards:
            out_name = f"{label}_{shard_path.stem}.parquet"
            out_path = str(staging_dir / out_name)
            if not os.path.exists(out_path):
                pairs.append((str(shard_path), out_path))
        return pairs

    book_pairs = build_pairs(book_shards, "book")
    news_pairs = build_pairs(news_shards, "news")
    all_pairs = book_pairs + news_pairs

    if not all_pairs:
        print("\nAll shards already filtered (staging dir has results). Skipping to resharding.")
    else:
        # Resume support
        n_already = (len(book_shards) + len(news_shards)) - len(all_pairs)
        if n_already > 0:
            print(f"\nResuming: {n_already} shards already filtered, {len(all_pairs)} remaining")

        # Group into batches
        random.shuffle(all_pairs)
        batches = []
        for i in range(0, len(all_pairs), args.shards_per_batch):
            batches.append(all_pairs[i:i + args.shards_per_batch])
        random.shuffle(batches)

        print(f"\nPhase 1: Filtering {len(all_pairs)} shards in {len(batches)} batches")
        print(f"  Workers: {args.workers}, Mem/worker: {args.mem_per_worker} GB")

        t0 = time.time()

        batch_jsons = [json.dumps(batch) for batch in batches]
        staging_dir_str = str(staging_dir)

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
                [staging_dir_str] * len(batches),
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

        print(f"\n{'='*60}")
        print(f"PHASE 1: FILTERING COMPLETE")
        print(f"{'='*60}")
        print(f"  Input docs:    {total_in:,}")
        print(f"  Kept:          {total_kept:,} ({100*total_kept/max(total_in,1):.1f}%)")
        print(f"  Rejected:      {total_rejected:,} ({100*total_rejected/max(total_in,1):.1f}%)")
        print(f"  Clean chars:   {total_chars/1e9:.2f}B (~{total_chars/CHARS_PER_TOKEN/1e9:.2f}B tokens)")
        if n_failed:
            print(f"  Failed batches: {n_failed}")
        print(f"  Elapsed:       {elapsed:.0f}s ({elapsed/3600:.2f}h)")

        if all_reasons:
            print(f"\n  Top rejection reasons:")
            for reason, count in sorted(all_reasons.items(), key=lambda x: -x[1])[:20]:
                print(f"    {reason}: {count:,}")

    # ----------------------------------------------------------------
    # 3. Phase 2: Collect, chunk, balance, and write output shards
    # ----------------------------------------------------------------
    # Goals:
    #   - All train shards have EQUAL token counts (via round-robin on sorted lengths)
    #   - Row groups per shard divisible by world_size for even DDP distribution
    #   - Docs chunked to max_chars so BOS-aligned dataloader wastes nothing
    # ----------------------------------------------------------------
    print(f"\nPhase 2: Collecting staged files, chunking, and resharding...")
    print(f"  Chunking: max_chars={args.max_chars}, min_chars={args.min_chars}")
    t1 = time.time()

    book_staged = sorted(staging_dir.glob("book_*.parquet"))
    news_staged = sorted(staging_dir.glob("news_*.parquet"))

    print(f"  Staged book files: {len(book_staged)}")
    print(f"  Staged news files: {len(news_staged)}")

    def collect_and_chunk(staged_files, target_chars, label):
        """Read staged parquets, chunk long docs, collect until target_chars."""
        chunks_out = []
        raw_docs = 0
        chunked_docs = 0
        chars = 0
        random.shuffle(staged_files)
        for sf in staged_files:
            if chars >= target_chars:
                break
            try:
                pf = pq.ParquetFile(sf)
                for rg_idx in range(pf.num_row_groups):
                    table = pf.read_row_group(rg_idx, columns=["text"])
                    batch = table.column("text").to_pylist()
                    del table
                    for text in batch:
                        if not text:
                            continue
                        raw_docs += 1
                        doc_chunks = chunk_document(text, max_chars=args.max_chars, min_chars=args.min_chars)
                        if len(doc_chunks) > 1:
                            chunked_docs += 1
                        for c in doc_chunks:
                            chunks_out.append(c)
                            chars += len(c)
                            if chars >= target_chars:
                                break
                        if chars >= target_chars:
                            break
                    if chars >= target_chars:
                        break
            except Exception:
                continue
            gc.collect()
        print(f"  {label}: {raw_docs:,} docs -> {len(chunks_out):,} chunks "
              f"({chunked_docs:,} split), {chars/1e9:.2f}B chars "
              f"(~{chars/CHARS_PER_TOKEN/1e9:.2f}B tokens)")
        return chunks_out

    book_chunks = collect_and_chunk(book_staged, target_book_chars, "Books")
    news_chunks = collect_and_chunk(news_staged, target_news_chars, "Newspapers")

    all_chunks = book_chunks + news_chunks
    del book_chunks, news_chunks
    total_chars = sum(len(t) for t in all_chunks)
    est_tokens = total_chars / CHARS_PER_TOKEN

    print(f"\n  Total: {len(all_chunks):,} chunks, {total_chars/1e9:.2f}B chars (~{est_tokens/1e9:.2f}B tokens)")

    # Shuffle before train/val split (ensures mix of books and newspapers)
    random.shuffle(all_chunks)

    # Train/val split
    n_val = max(1, int(len(all_chunks) * args.val_fraction))
    val_chunks = all_chunks[-n_val:]
    train_chunks = all_chunks[:-n_val]
    del all_chunks

    # -------------------------------------------------------------------
    # Compute docs_per_shard so that num_row_groups is divisible by world_size.
    # This ensures every GPU reads the exact same number of row groups per shard.
    #
    #   docs_per_shard = row_group_size * num_rg_per_gpu * world_size
    #
    # With row_group_size=500, world_size=8, num_rg_per_gpu=12:
    #   docs_per_shard = 500 * 12 * 8 = 48000  (96 row groups)
    # -------------------------------------------------------------------
    rgs_per_gpu = max(1, round(args.docs_per_shard / (args.row_group_size * args.world_size)))
    docs_per_shard = args.row_group_size * rgs_per_gpu * args.world_size
    n_rg_per_shard = docs_per_shard // args.row_group_size

    n_train_shards = max(1, len(train_chunks) // docs_per_shard)
    n_use = n_train_shards * docs_per_shard  # exact fit
    if n_use < len(train_chunks):
        # Put excess into val
        val_chunks = train_chunks[n_use:] + val_chunks
    train_chunks = train_chunks[:n_use]

    print(f"\n  Train: {len(train_chunks):,} chunks in {n_train_shards} shards")
    print(f"    {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups/shard "
          f"({rgs_per_gpu} per GPU x {args.world_size} GPUs)")
    print(f"  Val:   {len(val_chunks):,} chunks (1 shard)")

    # -------------------------------------------------------------------
    # Token-balanced distribution via round-robin on sorted lengths.
    #
    # Sort all train chunks by length, then deal them round-robin across
    # N shards. This ensures each shard gets an equal mix of short and long
    # chunks, so total chars (tokens) per shard are nearly identical.
    # -------------------------------------------------------------------
    print(f"\n  Balancing tokens across shards (sort + round-robin)...")
    train_chunks.sort(key=len)

    shard_buckets: list[list[str]] = [[] for _ in range(n_train_shards)]
    for i, chunk in enumerate(train_chunks):
        shard_buckets[i % n_train_shards].append(chunk)
    del train_chunks

    # Verify balance
    shard_char_counts = [sum(len(c) for c in bucket) for bucket in shard_buckets]
    avg_chars = sum(shard_char_counts) / len(shard_char_counts)
    min_chars_shard = min(shard_char_counts)
    max_chars_shard = max(shard_char_counts)
    imbalance = (max_chars_shard - min_chars_shard) / avg_chars * 100
    print(f"  Token balance: avg={avg_chars/CHARS_PER_TOKEN/1e6:.1f}M tok/shard, "
          f"min={min_chars_shard/CHARS_PER_TOKEN/1e6:.1f}M, "
          f"max={max_chars_shard/CHARS_PER_TOKEN/1e6:.1f}M, "
          f"imbalance={imbalance:.2f}%")

    # Write train shards
    print(f"\n  Writing {n_train_shards} train shards...")
    for shard_idx in range(n_train_shards):
        shard_docs = shard_buckets[shard_idx]
        random.shuffle(shard_docs)  # randomize order within shard for training
        out_path = args.output / f"shard_{shard_idx:05d}.parquet"
        table = pa.table({"text": shard_docs})
        pq.write_table(table, out_path,
                        row_group_size=args.row_group_size,
                        compression="snappy")
        del table
    del shard_buckets
    gc.collect()

    # Write val shard (always last, as expected by dataloader)
    out_idx = n_train_shards
    print(f"  Writing validation shard (shard_{out_idx:05d})...")
    val_path = args.output / f"shard_{out_idx:05d}.parquet"
    table = pa.table({"text": val_chunks})
    pq.write_table(table, val_path,
                    row_group_size=args.row_group_size,
                    compression="snappy")
    del table
    out_idx += 1

    elapsed2 = time.time() - t1

    # Report
    shard_sizes = []
    for i in range(n_train_shards):
        sp = args.output / f"shard_{i:05d}.parquet"
        if sp.exists():
            shard_sizes.append(os.path.getsize(sp))

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")
    print(f"  Train: {n_train_shards} shards, {n_use:,} chunks")
    print(f"    {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups (÷ {args.world_size} = {rgs_per_gpu} per GPU)")
    if shard_sizes:
        print(f"    File sizes: avg={sum(shard_sizes)/len(shard_sizes)/1e6:.1f}MB, "
              f"min={min(shard_sizes)/1e6:.1f}MB, max={max(shard_sizes)/1e6:.1f}MB")
    print(f"  Val:   1 shard, {len(val_chunks):,} chunks")
    print(f"  Total: {out_idx} shards")
    print(f"  Est. tokens: {est_tokens/1e9:.2f}B (effective, post-chunking)")
    print(f"  Output: {args.output}")
    print(f"  Phase 2 elapsed: {elapsed2:.0f}s")
    print()
    print(f"To use for training:")
    print(f"  ln -sfn {args.output.resolve()} /mnt/main0/data/michaelhla/gpt1900_training/base_data")


if __name__ == "__main__":
    main()
