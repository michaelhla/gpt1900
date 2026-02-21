#!/usr/bin/env python3
"""
Clean the ENTIRE pre-1900 filtered corpus and reshard into v3-compatible
format for training. Unlike sample_10b.py which samples a ~10B subset,
this keeps ALL clean documents (~27B tokens expected).

Two phases (can be run independently with --phase):
  Phase 1 (Dask-parallel): Each worker filters a batch of input shards and
      writes per-shard cleaned parquet files to a staging directory.
      Adaptive batching groups files by size for efficiency.
  Phase 2 (serial): Reads ALL staged files, chunks, balances, and writes
      final output shards in the format expected by the nanochat data loader.

Output format:
    shard_XXXXX.parquet  (text column only, snappy compression)
    Last shard = validation split

Usage:
    # Run both phases:
    python scripts/pre1900_scripts/clean_full_corpus.py

    # Run phases independently (for restartability):
    python scripts/pre1900_scripts/clean_full_corpus.py --phase filter
    python scripts/pre1900_scripts/clean_full_corpus.py --phase reshard
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
POST_1900_YEAR_RE = re.compile(
    r"\b(?:19\d{2}|20[0-2]\d)\b"  # matches 1900-2029
)
POST_1900_YEAR_THRESHOLD = 5

CHARS_PER_TOKEN = 4.2

# ============================================================================
# Document Chunking
# ============================================================================
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
# Adaptive batching by file size
# ============================================================================

def make_adaptive_batches(pairs: list[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Group (input_path, output_path) pairs into batches based on file size.

    Large files get small batches (or solo) to avoid OOM; small files get
    large batches to reduce Dask scheduling overhead.
    """
    # Annotate with file sizes
    sized = []
    for inp, out in pairs:
        try:
            sz = os.path.getsize(inp)
        except OSError:
            sz = 0
        sized.append((inp, out, sz))

    # Sort by size descending so large files are processed first
    sized.sort(key=lambda x: -x[2])

    batches: list[list[tuple[str, str]]] = []
    current_batch: list[tuple[str, str]] = []
    current_batch_size = 0

    for inp, out, sz in sized:
        # Determine max batch size based on file size
        if sz > 500_000_000:       # >500MB: solo
            max_per_batch = 1
        elif sz > 100_000_000:     # >100MB: 4/batch
            max_per_batch = 4
        elif sz > 10_000_000:      # >10MB: 16/batch
            max_per_batch = 16
        else:                      # <=10MB: 64/batch
            max_per_batch = 64

        # If adding this file would exceed the current batch's limit, flush
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
        description="Clean full pre-1900 corpus with post-1900 physics removal and reshard (Dask-parallelized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_filtered"),
                        help="Filtered data directory (with source subdirectories)")
    parser.add_argument("--output", "-o", type=Path, default=Path("/mnt/main0/data/michaelhla/pre1900_full_clean"),
                        help="Output directory for clean parquet shards")
    parser.add_argument("--phase", choices=["both", "filter", "reshard"], default="both",
                        help="Which phase to run (filter=Phase 1, reshard=Phase 2, both=both)")
    parser.add_argument("--docs-per-shard", type=int, default=48000,
                        help="Target documents per output shard (rounded to make row groups divisible by world-size)")
    parser.add_argument("--row-group-size", type=int, default=500,
                        help="Row group size within each parquet file")
    parser.add_argument("--val-fraction", type=float, default=0.005,
                        help="Fraction of data for validation (last shard)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Max characters per document chunk (long docs split at natural boundaries)")
    parser.add_argument("--min-chars", type=int, default=200,
                        help="Min characters per chunk (shorter chunks dropped)")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of Dask workers")
    parser.add_argument("--mem-per-worker", type=int, default=48,
                        help="Memory per worker in GB")
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

    # ================================================================
    # Phase 1: Dask-parallel filtering
    # ================================================================
    if args.phase in ("both", "filter"):
        print("=" * 60)
        print("PHASE 1: FILTERING ALL INPUT FILES")
        print("=" * 60)

        # Discover ALL files from all 4 source dirs
        source_dirs = ["blbooks", "books", "institutional_split", "newspapers"]

        all_files: list[tuple[Path, int]] = []
        for subdir_name in source_dirs:
            subdir = args.input / subdir_name
            if not subdir.is_dir():
                print(f"  Warning: {subdir} not found, skipping")
                continue
            count = 0
            for f in sorted(subdir.glob("*.parquet")):
                try:
                    sz = os.path.getsize(f)
                    all_files.append((f, sz))
                    count += 1
                except Exception:
                    pass
            print(f"  {subdir_name}: {count} files")

        total_size = sum(sz for _, sz in all_files)
        print(f"\n  Total: {len(all_files)} files, {total_size / 1e9:.1f} GB compressed")

        # Build (input, output) pairs, skipping already-staged files
        pairs: list[tuple[str, str]] = []
        n_already = 0
        for fpath, _ in all_files:
            # Use parent dir name as prefix to avoid collisions
            label = fpath.parent.name
            out_name = f"{label}_{fpath.stem}.parquet"
            out_path = str(staging_dir / out_name)
            if os.path.exists(out_path):
                n_already += 1
            else:
                pairs.append((str(fpath), out_path))

        if n_already > 0:
            print(f"\n  Resuming: {n_already} files already filtered, {len(pairs)} remaining")

        if not pairs:
            print("\n  All files already filtered (staging dir has results). Skipping to Phase 2.")
        else:
            # Adaptive batching by file size
            batches = make_adaptive_batches(pairs)
            random.shuffle(batches)

            print(f"\n  Filtering {len(pairs)} files in {len(batches)} adaptive batches")
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

            print(f"\n{'=' * 60}")
            print(f"PHASE 1 COMPLETE")
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
    # Phase 2: Collect, chunk, balance, and write output shards
    # ================================================================
    if args.phase in ("both", "reshard"):
        print(f"\n{'=' * 60}")
        print(f"PHASE 2: STREAMING CHUNK AND RESHARD")
        print(f"{'=' * 60}")
        print(f"  Chunking: max_chars={args.max_chars}, min_chars={args.min_chars}")
        t1 = time.time()

        staged_files = sorted(staging_dir.glob("*.parquet"))
        random.shuffle(staged_files)  # randomize source order
        print(f"  Staged files: {len(staged_files)}")

        if not staged_files:
            print("  ERROR: No staged files found. Run --phase filter first.")
            return

        # -------------------------------------------------------------------
        # Compute docs_per_shard so that num_row_groups is divisible by world_size.
        # -------------------------------------------------------------------
        rgs_per_gpu = max(1, round(args.docs_per_shard / (args.row_group_size * args.world_size)))
        docs_per_shard = args.row_group_size * rgs_per_gpu * args.world_size
        n_rg_per_shard = docs_per_shard // args.row_group_size

        print(f"  Target: {docs_per_shard} docs/shard, {n_rg_per_shard} row_groups/shard "
              f"({rgs_per_gpu} per GPU x {args.world_size} GPUs)")

        # -------------------------------------------------------------------
        # Streaming one-pass: read staged files, chunk, buffer N shards worth,
        # sort+round-robin for token balance, flush to disk. Reservoir sample
        # for validation throughout.
        # -------------------------------------------------------------------
        BUFFER_SHARDS = 10  # balance across 10 shards at a time (~2GB in memory)
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
            """Sort + round-robin distribute buffer into shards, write them.
            Returns number of shards written."""
            if not buf:
                return 0
            n_shards = len(buf) // docs_per_shard
            if n_shards == 0:
                return 0
            n_use = n_shards * docs_per_shard
            # Excess stays for next flush (returned via truncation)
            to_shard = buf[:n_use]

            # Token-balanced distribution: sort by length, round-robin
            to_shard.sort(key=len)
            buckets: list[list[str]] = [[] for _ in range(n_shards)]
            for i, chunk in enumerate(to_shard):
                buckets[i % n_shards].append(chunk)

            for j, bucket in enumerate(buckets):
                random.shuffle(bucket)  # randomize within shard
                out_path = args.output / f"shard_{start_shard + j:05d}.parquet"
                table = pa.table({"text": bucket})
                pq.write_table(table, out_path,
                               row_group_size=args.row_group_size,
                               compression="snappy")
                shard_char_counts.append(sum(len(c) for c in bucket))
                del table

            written = n_shards
            if (start_shard + written) % 50 < written or start_shard == 0:
                print(f"    Written shards {start_shard:,}-{start_shard + written - 1:,}")

            # Remove used chunks from buffer
            del buf[:n_use]
            gc.collect()
            return written

        print(f"\n  Streaming through staged files (buffer={BUFFER_SHARDS} shards)...")

        for file_idx, sf in enumerate(staged_files):
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
                            # Reservoir sample for validation
                            if random.random() < args.val_fraction:
                                val_reservoir.append(c)
                            else:
                                buffer.append(c)
            except Exception:
                continue

            # Flush when buffer is full
            if len(buffer) >= buffer_capacity:
                n_written = flush_buffer(buffer, shard_idx)
                shard_idx += n_written

            if (file_idx + 1) % 500 == 0:
                est_tok = total_chars / CHARS_PER_TOKEN
                print(f"    Processed {file_idx + 1}/{len(staged_files)} files, "
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
                        compression="snappy")
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
        print()
        print(f"To use for training:")
        print(f"  ln -sfn {args.output.resolve()} /mnt/main0/data/michaelhla/gpt1900_training/base_data")


if __name__ == "__main__":
    main()
