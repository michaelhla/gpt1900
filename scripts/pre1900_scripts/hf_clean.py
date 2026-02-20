#!/usr/bin/env python3
"""
Text Cleaning Script for Historical Corpus (Dask-parallelized).

Cleans and filters parquet shards from hf_download.py for LLM training.
Removes OCR artifacts, non-English content, boilerplate, library stamps,
and low-quality documents.

Usage:
    python scripts/pre1900_scripts/hf_clean.py \
        --input /mnt/main0/data/michaelhla/pre1900_raw \
        --output /mnt/main0/data/michaelhla/pre1900_clean \
        --workers 64
"""

from __future__ import annotations
import argparse
import re
import unicodedata
from pathlib import Path



# ============================================================================
# OCR Artifact Patterns
# ============================================================================

OCR_GARBAGE_PATTERNS = [
    r'[■□▪▫●○◆◇★☆♦♣♠♥]+',
    r'[\^~`]{2,}',
    r'[|!]{3,}',
    r'\.{5,}',
    r'\*{5,}',
    r'_{5,}',
    r'-{5,}',
    r'[^\x00-\x7F]{5,}',
    r'[A-Z\s]{50,}',
]


# ============================================================================
# Google Books Boilerplate Detection and Removal
# ============================================================================

GOOGLE_TEXT_BLOCKS = [
    r"Google's\s+mission\s+is\s+to\s+organize\s+the\s+world's\s+information.*?(?:Google\s+Book\s+Search|search\s+engine).*?\n",
    r'This\s+is\s+a\s+digital\s+copy\s+of\s+a\s+book\s+that\s+was\s+preserved.*?'
    r'(?:books\.google\.com|public\s+domain|Google\s+Book\s+Search).*?\n',
    r'About\s+Google\s+Book\s+Search.*?(?:books\.google\.com|Book\s+Search).*?\n',
    r'(?:Whether\s+a\s+book\s+is\s+in\s+the\s+public\s+domain|'
    r'public\s+domain\s+in\s+the\s+United\s+States).*?Google.*?\n',
    r'(?:We\s+encourage\s+the\s+use\s+of\s+public\s+domain|'
    r'Usage\s+guidelines|'
    r'Please\s+do\s+not\s+remove\s+this).*?Google.*?\n',
    r'(?:Maintain\s+attribution|Do\s+not\s+assume).*?(?:Google|public\s+domain).*?\n',
]

GOOGLE_LINE_PATTERNS = [
    r'^[^\n]*[Gg][Oo]{2}[Gg][Ll][Ee][^\n]*$',
    r'^[^\n]*books\s*\.?\s*google\s*\.?\s*com[^\n]*$',
    r'^[^\n]*Digitized\s+by[^\n]*$',
    r'^[^\n]*digital\s+copy\s+of\s+a\s+book[^\n]*$',
    r'^[^\n]*public\s+domain\s+in\s+the\s+United\s+States[^\n]*$',
]

GOOGLE_OCR_MANGLED = [
    r'^[^\n]{0,30}[Gg]\s*[Oo0]\s*[Oo0]\s*[Gg]\s*[Ll1]\s*[Ee3][^\n]{0,10}$',
    r'^\s*[^a-zA-Z\n]{0,15}[Gg]oogle\s*$',
    r'^\s*[DdLl][^\n]{0,20}[Gg]oogle\s*$',
    r'^\s*.*?igiti[sz]ed.*?[Gg]oogle.*$',
]

BOILERPLATE_LINE_PATTERNS = [
    r'^.*?legal\s+copyright\s+term\s+has\s+expired.*$',
    r'^.*?we\s+are\s+merely\s+their\s+custodians.*$',
    r'^.*?We\s+also\s+ask\s+that\s+you.*$',
    r'^.*?personal,?\s+non-commercial\s+purposes.*$',
    r'^.*?Keep\s+it\s+legal.*$',
    r'^.*?Copyright\s+infringement\s+liabil.*$',
    r'^.*?discover\s+the\s+world\'?s\s+books.*$',
    r'^.*?search\s+through\s+the\s+full\s+text.*$',
    r'^.*?placing\s+technical\s+restrictions.*$',
    r'^.*?automated\s+querying.*$',
    r'^.*?helping\s+authors\s+and\s+publishers.*$',
    r'^.*?reach\s+new\s+audiences.*$',
    r'^.*?Whether\s+a\s+book\s+is\s+still\s+in\s+copyright.*$',
    r'^.*?varies\s+from\s+country\s+to\s+country.*$',
    r'^.*?specific\s+use\s+of\s+any\s+specific\s+book.*$',
    r'^.*?anywhere\s+in\s+the\s+world.*$',
    r'^.*?public\s+domain\s+material.*$',
    r'^.*?Hosted\s+by\s*$',
    r'^.*?authorized\s+facsimile.*$',
    r'^.*?UNIVERSITY\s+MICROFILMS.*$',
    r'^.*?microfilm-xerography.*$',
    r'^.*?acid-free\s+paper.*$',
    r'^.*?Ann\s+Arbor,?\s+Michigan.*$',
    r'^.*?http\s*:\s*//books.*$',
    r'^.*?qoo[qg]le.*$',
    r'^.*?translation,?\s+optical\s+character\s+recognition.*$',
    r'^.*?access\s+to\s+a\s+large\s+amount\s+of\s+text.*$',
    r'^.*?please\s+contact\s+us.*$',
    r'^.*?IN\s+THE\s+CUSTODY\s+O[fF]\s+THE.*$',
    r'^.*?UNIVERSITY\s+of\s+CALIFO[RN]+IA.*$',
    r'^.*?LOS\s+ANGELES\s*$',
    r'^.*?LIBRARY\s*$',
]

GOOGLE_TEXT_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in GOOGLE_TEXT_BLOCKS]

# Combine line-based patterns into single alternation regexes for speed.
# Each group was 5-32 separate patterns doing full-text scans; merging into one
# regex means we scan the text once instead of N times.
_GOOGLE_LINE_COMBINED = '|'.join(GOOGLE_LINE_PATTERNS)
GOOGLE_LINE_COMPILED = re.compile(_GOOGLE_LINE_COMBINED, re.MULTILINE | re.IGNORECASE)

_GOOGLE_OCR_COMBINED = '|'.join(GOOGLE_OCR_MANGLED)
GOOGLE_OCR_COMPILED = re.compile(_GOOGLE_OCR_COMBINED, re.MULTILINE | re.IGNORECASE)

# Extract the keyword cores from ^.*?<keyword>.*$ patterns and combine
# into a single ^.*?(?:kw1|kw2|...).*$ alternation
_BOILERPLATE_KEYWORDS = [
    r'legal\s+copyright\s+term\s+has\s+expired',
    r'we\s+are\s+merely\s+their\s+custodians',
    r'We\s+also\s+ask\s+that\s+you',
    r'personal,?\s+non-commercial\s+purposes',
    r'Keep\s+it\s+legal',
    r'Copyright\s+infringement\s+liabil',
    r"discover\s+the\s+world'?s\s+books",
    r'search\s+through\s+the\s+full\s+text',
    r'placing\s+technical\s+restrictions',
    r'automated\s+querying',
    r'helping\s+authors\s+and\s+publishers',
    r'reach\s+new\s+audiences',
    r'Whether\s+a\s+book\s+is\s+still\s+in\s+copyright',
    r'varies\s+from\s+country\s+to\s+country',
    r'specific\s+use\s+of\s+any\s+specific\s+book',
    r'anywhere\s+in\s+the\s+world',
    r'public\s+domain\s+material',
    r'Hosted\s+by\s*$',
    r'authorized\s+facsimile',
    r'UNIVERSITY\s+MICROFILMS',
    r'microfilm-xerography',
    r'acid-free\s+paper',
    r'Ann\s+Arbor,?\s+Michigan',
    r'http\s*:\s*//books',
    r'qoo[qg]le',
    r'translation,?\s+optical\s+character\s+recognition',
    r'access\s+to\s+a\s+large\s+amount\s+of\s+text',
    r'please\s+contact\s+us',
    r'IN\s+THE\s+CUSTODY\s+O[fF]\s+THE',
    r'UNIVERSITY\s+of\s+CALIFO[RN]+IA',
    r'LOS\s+ANGELES\s*$',
    r'LIBRARY\s*$',
]
BOILERPLATE_LINE_COMPILED = re.compile(
    r'^.*?(?:' + '|'.join(_BOILERPLATE_KEYWORDS) + r').*$',
    re.MULTILINE | re.IGNORECASE,
)


# ============================================================================
# HathiTrust / British Library Boilerplate
# ============================================================================

HATHI_PATTERNS = [
    r'(?i)HathiTrust\s+Digital\s+Library[^\n]*\n',
    r'(?i)Generated\s+.*?HathiTrust[^\n]*\n',
    r'(?i)Public\s+Domain\s+.*?Google-digitized[^\n]*\n',
]

# ============================================================================
# Library stamps and headers
# ============================================================================

LIBRARY_PATTERNS = [
    r'(?i)university\s+of\s+\w+\s*[•·]\s*\w+',
    r'(?i)the\s+library\s+of\s+the\s+university',
    r'(?i)(?:bequest|gift)\s+of\s+[A-Z][^\n]{0,50}\n',
    r'(?i)public\s+domain.*?google',
    r'(?i)scanned\s+by\s+[^\n]{0,50}\n',
    r'(?i)from\s+the\s+collections?\s+of[^\n]*\n',
]


# ============================================================================
# Text Reflow Patterns
# ============================================================================

# Matches a word fragment (2+ lowercase letters) followed by a hyphen at
# end of line, then a lowercase letter starting the next line. Catches
# line-break hyphenation like "collect-\ning" but not em-dashes or
# real compound words mid-line.
HYPHEN_LINEBREAK_RE = re.compile(r'([a-z]{2,})-\n([a-z])')


# ============================================================================
# Project Gutenberg markers
# ============================================================================

PG_START = re.compile(r'\*{3}\s*START\s+OF\s+.*?PROJECT\s+GUTENBERG.*?\*{3}', re.IGNORECASE | re.DOTALL)
PG_END = re.compile(r'\*{3}\s*END\s+OF\s+.*?PROJECT\s+GUTENBERG.*?\*{3}', re.IGNORECASE | re.DOTALL)


# ============================================================================
# Post-1900 Physics Anachronism Detection
# ============================================================================
# Detects references to physics concepts that didn't exist before 1900,
# which would indicate data leakage (e.g., modern annotations, introductions)
#
# Key dates for reference:
#   - photon: term popularized 1926 (G.N. Lewis)
#   - isotope: coined 1913
#   - proton: named by Rutherford 1920
#   - neutron: discovered 1932
#   - spacetime: Minkowski framework 1908
#   - uncertainty principle: 1927
#   - wave function: Schrödinger 1926
#   - dark matter: term coined 1933
#   - dark energy: term coined 1998
#   - event horizon: term coined 1950s
#   - mass-energy equivalence: 1905+
#   - time dilation: 1905+
#
# Note on ambiguous terms:
#   - "quantum" is a general Latin word ("amount") used in law/logic pre-1900
#   - "relativity" was used in philosophy long before Einstein
#   - "nuclear" meant "relating to a nucleus" since 1822
#   - Planck/Rutherford published in 1890s, so bare names may appear in pre-1900 refs

# Tier 1: Single match triggers rejection
# These are highly specific post-1900 physics terms/phrases with no pre-1900 meaning
TIER1_EXPLICIT_PHRASES = [
    # Specific physics terminology (no pre-1900 usage)
    r"\bphoton(?:s|ic)?\b",                    # 1926
    r"\bisotope(?:s|ic)?\b",                   # 1913
    r"\bproton(?:s)?\b",                       # 1920 (as particle name)
    r"\bneutron(?:s)?\b",                      # 1932
    r"\bspacetime\b",                          # 1908
    r"\bspace-time\b",                         # 1908
    r"\bwave\s+function(?:s)?\b",              # 1926
    r"\buncertainty\s+principle\b",            # 1927
    r"\bdark\s+matter\b",                      # 1933
    r"\bdark\s+energy\b",                      # 1998
    r"\bevent\s+horizon(?:s)?\b",              # 1950s
    r"\bmass[–-]energy\b",                     # 1905+ (relativity concept)
    r"\btime\s+dilation\b",                    # 1905+
    r"\bradioactive\s+decay\b",                # post-1900 terminology

    # Explicit modern physics phrases
    r"einstein'?s?\s+(?:theory|equation|relativity|field)",
    r"theory\s+of\s+(?:special|general)\s+relativity",
    r"e\s*=\s*mc\s*(?:\^|²|2)",
    r"quantum\s+mechanics",
    r"quantum\s+field\s+theory",
    r"quantum\s+electrodynamics",
    r"heisenberg'?s?\s+uncertainty",
    r"schr[öo]dinger'?s?\s+(?:equation|cat|wave)",
    r"bohr\s+(?:model|radius|atom)",
    r"rutherford'?s?\s+(?:model|experiment|scattering)",
    r"planck(?:'s)?\s+(?:constant|law|length|time|mass)",
    r"nuclear\s+(?:fission|fusion|reactor|bomb|weapon|physics|reaction|energy)",
    r"atomic\s+bomb",
    r"big\s+bang(?:\s+theory)?",
    r"expanding\s+universe",
    r"black\s+hole(?:s)?",
    r"particle\s+accelerator(?:s)?",
    r"standard\s+model",                       # particle physics
    r"higgs\s+(?:boson|field|particle)",
    r"antimatter",
    r"positron(?:s)?",                         # discovered 1932
    r"quark(?:s)?",                            # 1964
    r"lepton(?:s)?",                           # modern particle physics
    r"fermion(?:s)?",
    r"boson(?:s)?",
]

# Tier 2: Context-based patterns - need 2+ matches to reject
# These terms are ambiguous on their own but suspicious in physics context
TIER2_CONTEXT_PATTERNS = [
    # "relativity" needs physics context (used in philosophy pre-1900)
    r"\b(?:special|general)\s+relativity\b",
    r"\btheory\s+of\s+relativity\b",
    r"\brelativistic\b",                       # clearly physics usage

    # "quantum" needs physics context (Latin "amount" used in law pre-1900)
    r"\bquantum\s+(?:theory|state|field|number|physics|level|effect)\b",
    r"\bquanta?\s+of\s+(?:energy|light|radiation|action)\b",
    r"\bquantum\b(?=.*\b(?:planck|photon|electron|atom)\b)",
    r"\bquantized\b",
    r"\bquantization\b",

    # "nuclear" needs physics context (generic "nucleus" usage since 1822)
    r"\bnuclear\s+(?:force|decay|radiation|spin|magnetic)\b",

    # Half-life in radioactive context
    r"\bhalf[- ]life\b(?=.*\b(?:decay|radioactive|isotope|nuclide|uranium|radium)\b)",

    # Physicist names with concept (names alone may appear pre-1900)
    r"\beinstein\b(?=.*\b(?:relativity|photon|photoelectric|brownian)\b)",
    r"\bheisenberg\b(?=.*\b(?:uncertainty|matrix|quantum)\b)",
]

TIER1_COMPILED = [re.compile(p, re.IGNORECASE) for p in TIER1_EXPLICIT_PHRASES]
TIER2_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in TIER2_CONTEXT_PATTERNS]


def estimate_printable_ratio(text: str, sample_size: int = 10000) -> float:
    """Estimate ratio of printable ASCII characters."""
    sample = text[:sample_size]
    if not sample:
        return 0.0

    printable = sum(1 for c in sample if c.isprintable() or c in '\n\t')
    return printable / len(sample)


def count_ocr_artifacts(text: str) -> int:
    """Count OCR artifact patterns in text."""
    count = 0
    sample = text[:30000]
    for pattern in OCR_GARBAGE_PATTERNS:
        count += len(re.findall(pattern, sample))
    return count


def contains_post_1900_physics(text: str) -> tuple[bool, str | None]:
    """
    Check if text contains anachronistic post-1900 physics concepts.

    Uses two-tier system:
    - Tier 1: Highly specific post-1900 terms (photon, isotope, etc.) - single match rejects
    - Tier 2: Context-based patterns for ambiguous terms (quantum, relativity, etc.) - need 2+ matches

    Returns (contains_anachronism, reason or None)
    """
    # Sample from text (beginning + end for efficiency on large docs)
    sample = text[:50000] + text[-50000:] if len(text) > 100000 else text

    # Check Tier 1 - any single match rejects
    for pattern in TIER1_COMPILED:
        match = pattern.search(sample)
        if match:
            return True, f"explicit: {match.group()}"

    # Check Tier 2 - need 2+ matches
    tier2_matches = []
    for pattern in TIER2_COMPILED:
        match = pattern.search(sample)
        if match:
            tier2_matches.append(match.group())
            if len(tier2_matches) >= 2:
                return True, f"multiple: {tier2_matches[0]}, {tier2_matches[1]}"

    return False, None


# ============================================================================
# Text Cleaning Functions
# ============================================================================

_BOILERPLATE_SIMPLE_KEYWORDS = [
    'copyright term has expired', 'merely their custodians',
    'we also ask that you', 'non-commercial purposes',
    'keep it legal', 'copyright infringement',
    "discover the world", 'search through the full text',
    'placing technical restrictions', 'automated querying',
    'helping authors and publishers', 'reach new audiences',
    'whether a book is still in copyright', 'varies from country to country',
    'specific use of any specific book', 'anywhere in the world',
    'public domain material', 'hosted by',
    'authorized facsimile', 'university microfilms',
    'microfilm-xerography', 'acid-free paper',
    'ann arbor', 'qooqle', 'qoogle',
    'optical character recognition', 'large amount of text',
    'please contact us', 'in the custody of the',
    'university of california', 'los angeles',
    'digitized by', 'digital copy of a book',
    'public domain in the united states', 'books.google',
]


def remove_google_boilerplate(text: str) -> str:
    """Remove Google Books boilerplate comprehensively."""
    # Multi-line block patterns (few, expensive but necessary)
    for pattern in GOOGLE_TEXT_COMPILED:
        text = pattern.sub('', text)

    # Fast line-by-line scan instead of 30+ regex passes
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        line_lower = line.lower()

        # Google line check
        if 'google' in line_lower:
            if len(line) < 100 or line_lower.find('google') < 50 or line_lower.find('google') > len(line) - 50:
                continue

        # OCR-mangled Google (G o o g l e with spaces)
        if GOOGLE_OCR_COMPILED.match(line):
            continue

        # Boilerplate keyword check — simple substring matching
        if any(kw in line_lower for kw in _BOILERPLATE_SIMPLE_KEYWORDS):
            continue

        # Standalone "LIBRARY" line
        if line.strip() == 'LIBRARY':
            continue

        filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def remove_hathi_boilerplate(text: str) -> str:
    """Remove HathiTrust boilerplate."""
    for pattern in HATHI_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text


def remove_library_stamps(text: str) -> str:
    """Remove library stamps and headers."""
    for pattern in LIBRARY_PATTERNS:
        text = re.sub(pattern, '', text)
    return text


def strip_pg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header/footer if present."""
    start_match = PG_START.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = PG_END.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters."""
    text = unicodedata.normalize('NFKC', text)

    replacements = {
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        'æ': 'ae', 'œ': 'oe', 'ſ': 's',
        '—': '--', '–': '-',
        '"': '"', '"': '"', ''': "'", ''': "'",
        '…': '...', '­': '',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace while preserving paragraph structure."""
    text = text.replace('\t', ' ')
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_ocr_artifacts(text: str) -> str:
    """Remove common OCR artifacts."""
    text = re.sub(r'^\s*[-\[]*\s*\d{1,4}\s*[-\]]*\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[a-zA-Z]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[^a-zA-Z]*$', '', text, flags=re.MULTILINE)
    return text


def rejoin_hyphenated_words(text: str) -> str:
    """Rejoin words hyphenated across line breaks: 'collect-\\ning' -> 'collecting'."""
    return HYPHEN_LINEBREAK_RE.sub(r'\1\2', text)


def reflow_text(text: str) -> str:
    """Replace single newlines with spaces, preserving paragraph breaks (\\n\\n).

    Undoes column-width line wrapping from books and newspaper column layouts.
    After this, each paragraph is one long continuous line.
    """
    paragraphs = text.split('\n\n')
    reflowed = []
    for para in paragraphs:
        line = para.replace('\n', ' ')
        line = re.sub(r'  +', ' ', line)
        reflowed.append(line.strip())
    return '\n\n'.join(reflowed)


def strip_leading_garbage(text: str, max_lines: int = 100) -> str:
    """Strip short garbage lines from the beginning of text."""
    # Patterns anchored to start of line (^)
    library_stamp_patterns = [
        # University / library names (standalone or as labels)
        r'(?i)^university\s+of', r'(?i)^the\s+library', r'(?i)^public\s+library',
        r'(?i)^boston\s+public', r'(?i)^property\s+of', r'(?i)^in\s+the\s+custody',
        r'(?i)^custody\s+o[fr]', r'(?i)^n\s+the\s+custody', r'(?i)^shelf\s+n',
        r'(?i)^accession', r'(?i)^call\s+n', r'(?i)^from\s+the\s+collection',
        r'(?i)^gift\s+of', r'(?i)^bequest\s+of', r'(?i)^presented\s+by',
        r'(?i)^digitized\s+by', r'(?i)^scanned\s+by', r'(?i)^oxford\s*$',
        r'(?i)^cambridge\s*$', r'(?i)^harvard\s*$', r'(?i)^yale\s*$',
        r'(?i)^cornell\s*$', r'(?i)^in\s+\d{4}\s+with\s+funding',
        r'(?i)^http://', r'(?i)^www\.', r'(?i)^archive\.org',
        # Harvard-specific stamps and labels
        r'(?i)^harvard\s+college', r'(?i)^harvard\s+university',
        r'(?i)^godfrey\s+lowell\s+cabot', r'(?i)^science\s+library',
        r'(?i)^science\s+center\s+library', r'(?i)^library\s+science\s+center',
        r'(?i)^cabot\s+science', r'(?i)^transferred\s+to',
        r'(?i)^from\s+the\s+library\s+of', r'(?i)^the\s+gift\s+of',
        r'(?i)^from\s+the\s+books\s+in', r'(?i)^from\s+the\s+bequest',
        r'(?i)^from\s+the\s+fund\s+of', r'(?i)^from\s+the\s+bright',
        r'(?i)^this\s+book\s+was\s+stolen', r'(?i)^it\s+was\s+later\s+recovered',
        r'(?i)^widener\s+library', r'(?i)^pickman\s+bequest',
        r'(?i)^bright\s+legacy', r'(?i)^\(?class\s+of\s+\d{4}',
        r'(?i)^received', r'(?i)^exlibris', r'(?i)^ex\s*libris',
        # Seal OCR fragments (Harvard/institutional seals) — anchored
        r'(?i)^sigill', r'(?i)^veri\s*tas', r'(?i)^ecclesia',
        r'(?i)^nov[\s:]+angl', r'(?i)^hristo', r'(?i)^christo',
        r'(?i)^harvardian', r'(?i)^academiae', r'(?i)^ademiae',
        r'(?i)^ardianae', r'(?i)^virginibus',
        # Call numbers (e.g. "Math 3208.59", "QA 211 T6 1880", "7701")
        r'^[A-Z]{1,4}\s+\d{1,5}[\s.]', r'^\d{3,7}\s*$',
        # Other institutional stamps
        r'(?i)^bodleian', r'(?i)^british\s+museum', r'(?i)^british\s+library',
        r'(?i)^national\s+library', r'(?i)^library\s+of\s+congress',
        r'(?i)^royal\s+library', r'(?i)^free\s+library',
        r'(?i)^lending\s+library', r'(?i)^circulating\s+library',
        r'(?i)^donated\s+by', r'(?i)^purchased\s+by',
        r'(?i)^entered\s+according\s+to\s+act',
    ]
    stamp_compiled = [re.compile(p) for p in library_stamp_patterns]

    # Keyword patterns that match ANYWHERE in a line (for compound lines like
    # "5078.29.3 VERI TAS HARVARD COLLEGE LIBRARY" where call number + stamp
    # are OCR'd onto the same line)
    stamp_keyword_patterns = [
        r'(?i)\bharvard\b',  # "Harvard" anywhere — rare in actual pre-1900 prose
        r'(?i)college\s+library', r'(?i)university\s+library',
        r'(?i)ve\w{0,2}r?i?\s*[ts]\w?as',  # VERI TAS, VERO TAS, VECRI TAS, VERI STAS, etc.
        r'(?i)harvardian', r'(?i)academiae',
        r'(?i)sigill\b', r'(?i)eccles\w{0,3}\b',  # ECCLESIA, ECCLESIE, ECCLESLE (OCR)
        r'(?i)\bbequest\b', r'(?i)\bgift\s+of\b',
        r'(?i)\btransferred\s+to\b', r'(?i)\bclass\s+of\s+\d{4}\b',
    ]
    stamp_keyword_compiled = [re.compile(p) for p in stamp_keyword_patterns]

    lines = text.split('\n')
    start_idx = 0

    for i, line in enumerate(lines[:max_lines]):
        stripped = line.strip()
        if len(stripped) == 0:
            continue

        # Check start-anchored patterns
        is_stamp = any(p.match(stripped) for p in stamp_compiled)
        # Check keyword patterns (anywhere in line) — only for short lines
        # to avoid false positives on real content that mentions "Harvard College"
        if not is_stamp and len(stripped) < 80:
            is_stamp = any(p.search(stripped) for p in stamp_keyword_compiled)
        if is_stamp:
            continue

        alpha_count = sum(1 for c in stripped if c.isalpha())
        alpha_ratio = alpha_count / max(len(stripped), 1)
        word_count = len(stripped.split())

        is_content = (
            (len(stripped) > 40 and alpha_ratio > 0.5) or
            (alpha_ratio > 0.8 and word_count >= 3 and len(stripped) > 15) or
            (word_count >= 4 and alpha_ratio > 0.6)
        )

        if is_content:
            start_idx = i
            break

    if start_idx > 0:
        return '\n'.join(lines[start_idx:])
    return text


def remove_front_matter(text: str, max_scan: int = 15000) -> str:
    """Try to find where the actual content starts."""
    text = strip_leading_garbage(text)

    content_markers = [
        r'(?i)^CHAPTER\s+[IVX\d]+', r'(?i)^BOOK\s+[IVX\d]+', r'(?i)^PART\s+[IVX\d]+',
        r'(?i)^INTRODUCTION\s*$', r'(?i)^PREFACE\s*\.?\s*$', r'(?i)^CONTENTS\s*$',
        r'(?i)^T\s*H\s*E\s*$', r'(?i)^THE\s+PREFACE', r'(?i)^TO\s+THE\s+READER',
        r'(?i)^ADVERTISEMENT\s*$', r'(?i)^DEDICATION\s*$', r'(?i)^PROLOGUE\s*$',
    ]

    front_section = text[:max_scan]
    earliest_pos = max_scan

    for pattern in content_markers:
        match = re.search(pattern, front_section, re.MULTILINE)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()

    if 200 < earliest_pos < max_scan:
        return text[earliest_pos:]

    return text


def clean_text(text: str) -> str:
    """Apply all cleaning steps to text."""
    text = remove_google_boilerplate(text)
    text = remove_hathi_boilerplate(text)
    text = remove_library_stamps(text)
    text = strip_pg_boilerplate(text)
    text = normalize_unicode(text)
    text = clean_ocr_artifacts(text)
    text = normalize_whitespace(text)
    text = rejoin_hyphenated_words(text)
    text = reflow_text(text)
    text = remove_front_matter(text)
    return text.strip()


# ============================================================================
# Document Processing
# ============================================================================

def normalize_ocr_score(ocr_score: float, source: str) -> float:
    """Normalize OCR score to 0-1 range.

    Institutional books store scores on 0-100 scale.
    Returns -1.0 for missing/unavailable scores.
    """
    if ocr_score < 0:
        return -1.0
    if source == "institutional-books" and ocr_score > 1.0:
        return ocr_score / 100.0
    return ocr_score


# Per-source OCR thresholds (normalized to 0-1).
# These are based on empirical analysis of text quality at each score level:
#   - institutional-books: below 90/100 is mostly tables, indices, math notation
#   - AmericanStories newspapers: legibility == 1.0 required (separate threshold)
SOURCE_OCR_THRESHOLDS = {
    "institutional-books": 0.90,
}
DEFAULT_LEGIBILITY_THRESHOLD = 1.0


def process_document(
    text: str,
    min_english: float,
    min_printable: float,
    max_ocr_artifacts: int,
    min_length: int,
    ocr_score: float = -1.0,
    legibility: float = -1.0,
    source: str = "",
    min_ocr_score: float | None = None,
    min_legibility: float | None = None,
) -> tuple[str | None, str]:
    """
    Process a single document.

    Args:
        ocr_score: Normalized OCR confidence (0-1) for books, -1 if unavailable.
        legibility: Legibility score (0-1) for newspapers, -1 if unavailable.
        source: Data source name (for per-source OCR thresholds).
        min_ocr_score: Override per-source OCR threshold. None = use SOURCE_OCR_THRESHOLDS.
        min_legibility: Override default legibility threshold. None = use DEFAULT_LEGIBILITY_THRESHOLD.

    Returns (cleaned_text or None, reason)
    """
    if len(text) < min_length:
        return None, f"too short ({len(text)} chars)"

    # OCR score filter (books) -- skip docs with a known-bad score, keep unknowns (-1)
    if ocr_score >= 0:
        thresh = min_ocr_score if min_ocr_score is not None else SOURCE_OCR_THRESHOLDS.get(source, 0.0)
        if thresh > 0 and ocr_score < thresh:
            return None, f"low OCR score ({ocr_score:.3f} < {thresh:.2f})"

    # Legibility filter (newspapers)
    if legibility >= 0:
        thresh = min_legibility if min_legibility is not None else DEFAULT_LEGIBILITY_THRESHOLD
        if thresh > 0 and legibility < thresh:
            return None, f"low legibility ({legibility:.3f} < {thresh:.2f})"

    printable_ratio = estimate_printable_ratio(text)
    if printable_ratio < min_printable:
        return None, f"low printable ratio ({printable_ratio:.2f})"

    # English ratio check (disabled by default since datasets are already English)
    if min_english > 0:
        english_ratio = estimate_english_ratio(text)
        if english_ratio < min_english:
            return None, f"low English ratio ({english_ratio:.2f})"

    artifact_count = count_ocr_artifacts(text)
    if artifact_count > max_ocr_artifacts:
        return None, f"too many OCR artifacts ({artifact_count})"

    # Check for post-1900 physics anachronisms (data leakage prevention)
    has_anachronism, reason = contains_post_1900_physics(text)
    if has_anachronism:
        return None, f"post-1900 physics ({reason})"

    cleaned = clean_text(text)

    if len(cleaned) < min_length:
        return None, f"too short after cleaning ({len(cleaned)} chars)"

    # Post-cleaning English check (also disabled by default)
    if min_english > 0:
        cleaned_english = estimate_english_ratio(cleaned)
        if cleaned_english < min_english * 0.9:
            return None, f"low English after cleaning ({cleaned_english:.2f})"

    return cleaned, "ok"


def process_shard_batch(
    batch_json: str,
    scripts_dir: str,
    min_english: float,
    min_printable: float,
    max_ocr_artifacts: int,
    min_length: int,
    min_ocr_score: float,
    min_legibility: float,
) -> dict:
    """
    Process a batch of (input_path, output_path) pairs.

    Imports hf_clean functions inside worker to avoid Dask serialization issues.
    Uses JSON-serialized batch list to avoid Dask nested-list serialization issues.

    Returns dict with stats: n_in, n_kept, n_failed, filter_reasons.
    """
    import json
    import os
    import sys
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Add scripts dir to path so we can import hf_clean
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from hf_clean import process_document, normalize_ocr_score

    pairs = json.loads(batch_json)

    total_in = 0
    total_kept = 0
    total_failed = 0
    filter_reasons: dict[str, int] = {}

    for input_path, output_path in pairs:
        try:
            table = pq.read_table(input_path)
        except Exception:
            continue

        texts = table.column("text").to_pylist()
        sources = table.column("source").to_pylist() if "source" in table.column_names else [""] * len(texts)
        doc_ids = table.column("doc_id").to_pylist() if "doc_id" in table.column_names else [""] * len(texts)
        years = table.column("year").to_pylist() if "year" in table.column_names else [0] * len(texts)
        ocr_scores = table.column("ocr_score").to_pylist() if "ocr_score" in table.column_names else [-1.0] * len(texts)
        legibilities = table.column("legibility").to_pylist() if "legibility" in table.column_names else [-1.0] * len(texts)

        output_rows: list[dict] = []

        for text, source, doc_id, year, raw_ocr, legibility in zip(
            texts, sources, doc_ids, years, ocr_scores, legibilities
        ):
            total_in += 1

            ocr_score = normalize_ocr_score(raw_ocr if raw_ocr is not None else -1.0, source)

            cleaned, reason = process_document(
                text,
                min_english,
                min_printable,
                max_ocr_artifacts,
                min_length,
                ocr_score=ocr_score,
                legibility=legibility if legibility is not None else -1.0,
                source=source,
                min_ocr_score=min_ocr_score if min_ocr_score >= 0 else None,
                min_legibility=min_legibility if min_legibility >= 0 else None,
            )

            if cleaned is not None:
                total_kept += 1
                output_rows.append({
                    "text": cleaned,
                    "source": source,
                    "doc_id": doc_id,
                    "year": year,
                    "ocr_score": ocr_score,
                    "legibility": legibility if legibility is not None else -1.0,
                })
            else:
                total_failed += 1
                bucket = reason.split("(")[0].strip()
                filter_reasons[bucket] = filter_reasons.get(bucket, 0) + 1

        if output_rows:
            columns: dict[str, list] = {}
            for row in output_rows:
                for k, v in row.items():
                    columns.setdefault(k, []).append(v)
            out_table = pa.table(columns)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pq.write_table(out_table, output_path, compression="snappy")

    return {
        "n_in": total_in,
        "n_kept": total_kept,
        "n_failed": total_failed,
        "filter_reasons": filter_reasons,
    }


def main():
    import glob
    import json
    import os
    import time

    from evolutionaryscale.utils.executor import DaskExecutorContext, TaskGranularity

    parser = argparse.ArgumentParser(
        description="Clean and filter historical text corpus from parquet shards (Dask-parallelized)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", type=Path, required=True,
                        help="Input directory with parquet shards")
    parser.add_argument("--output", "-o", type=Path, required=True,
                        help="Output directory for cleaned parquet shards")
    parser.add_argument("--min-english", type=float, default=0.0,
                        help="Minimum English word ratio (0-1). Set to 0 to disable.")
    parser.add_argument("--min-printable", type=float, default=0.85,
                        help="Minimum printable character ratio (0-1)")
    parser.add_argument("--max-ocr-artifacts", type=int, default=50,
                        help="Maximum OCR artifact patterns allowed")
    parser.add_argument("--min-length", type=int, default=5000,
                        help="Minimum text length in characters")
    parser.add_argument("--min-ocr-score", type=float, default=-1.0,
                        help="Override per-source OCR score threshold (0-1). "
                             "Default: -1 (use per-source defaults: 0.90 for institutional-books).")
    parser.add_argument("--min-legibility", type=float, default=-1.0,
                        help="Minimum legibility score for newspapers (0-1). Default: -1 (use default 1.0).")
    parser.add_argument("--workers", type=int, default=64,
                        help="Number of Dask workers")
    parser.add_argument("--shards-per-batch", type=int, default=64,
                        help="Number of input shards per worker batch")
    parser.add_argument("--mem-per-worker", type=int, default=16,
                        help="Memory per worker in GB")
    parser.add_argument("--partition", type=str, default="midpri",
                        help="Slurm partition")
    args = parser.parse_args()

    # Find all parquet files recursively
    input_files = sorted(glob.glob(str(args.input / "**" / "*.parquet"), recursive=True))

    if not input_files:
        print(f"No .parquet files found in {args.input}")
        return

    print(f"Found {len(input_files)} input shards")

    # Setup output — mirror input directory structure
    args.output.mkdir(parents=True, exist_ok=True)

    # Build (input_path, output_path) pairs, preserving subdirectory structure
    all_pairs = []
    for fpath in input_files:
        rel = os.path.relpath(fpath, args.input)
        out_path = str(args.output / rel)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        all_pairs.append((fpath, out_path))

    # Resume support: skip already-processed shards
    remaining = [(inp, out) for inp, out in all_pairs if not os.path.exists(out)]
    if len(remaining) < len(all_pairs):
        print(f"Resuming: {len(all_pairs) - len(remaining)} shards already processed, {len(remaining)} remaining")

    if not remaining:
        print("All shards already processed!")
        return

    # Group into batches
    batches = []
    for i in range(0, len(remaining), args.shards_per_batch):
        batches.append(remaining[i:i + args.shards_per_batch])

    n_shards = len(remaining)
    print(f"Processing {n_shards} shards in {len(batches)} batches ({args.shards_per_batch} shards/batch)")
    print(f"Using {args.workers} workers")

    t0 = time.time()

    # Serialize batches as JSON strings
    batch_jsons = [json.dumps(batch) for batch in batches]

    # Scripts dir for worker imports
    scripts_dir = str(Path(__file__).resolve().parent)

    with DaskExecutorContext(
        task_granularity=TaskGranularity.CPU,
        slurm_partition=args.partition,
        num_jobs=min(args.workers, len(batches)),
        cpus_per_worker=1,
        mem_per_worker_gb=args.mem_per_worker,
    ) as executor:
        results = executor.map(
            process_shard_batch,
            batch_jsons,
            [scripts_dir] * len(batches),
            [args.min_english] * len(batches),
            [args.min_printable] * len(batches),
            [args.max_ocr_artifacts] * len(batches),
            [args.min_length] * len(batches),
            [args.min_ocr_score] * len(batches),
            [args.min_legibility] * len(batches),
            progress=True,
            errors="skip",
        )

    # Aggregate stats
    total_in = 0
    total_kept = 0
    total_failed = 0
    n_batch_failed = 0
    filter_reasons: dict[str, int] = {}

    for result in results:
        if result is None:
            n_batch_failed += 1
            continue
        total_in += result["n_in"]
        total_kept += result["n_kept"]
        total_failed += result["n_failed"]
        for reason, count in result["filter_reasons"].items():
            filter_reasons[reason] = filter_reasons.get(reason, 0) + count

    elapsed = time.time() - t0

    print()
    print("=" * 60)
    print("CLEANING RESULTS")
    print("=" * 60)
    print(f"  Input docs:     {total_in:,}")
    print(f"  Kept:           {total_kept:,} ({100*total_kept/max(total_in,1):.1f}%)")
    print(f"  Filtered:       {total_failed:,} ({100*total_failed/max(total_in,1):.1f}%)")
    if n_batch_failed:
        print(f"  Failed batches: {n_batch_failed}")
    print(f"  Output dir:     {args.output}")
    print(f"  Elapsed:        {elapsed:.1f}s ({elapsed/3600:.2f}h)")

    if filter_reasons:
        print(f"\n  Filter reasons:")
        for reason, count in sorted(filter_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count:,}")


if __name__ == "__main__":
    main()
