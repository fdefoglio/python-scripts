#!/usr/bin/env python3
"""
bilingual_align.py — Align two bilingual texts sentence by sentence.

Reads from Input.txt in the same directory as the script.
Input.txt format:

    SourceLanguageName
    [source language text]
    ===
    TargetLanguageName
    [target language text]

Example:

    English
    It sought refuge in the past, myth, dreams, the supernatural, and the irrational.
    As the new political concept of "nation" emerged, Romantics regarded "common folk"...

    ===
    Afrikaans
    Dit het skuiling gesoek in die verlede, mite, drome, die bonatuurlike en die irrationele.
    Toe die nuwe politieke koncept van "nasie" na vore gekom het...

Outputs aligned pairs as both CSV and JSON.
"""

import re
import json
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Set
from itertools import zip_longest

# --------------------------------------------------------------
#  CONFIGURATION
# --------------------------------------------------------------

# Align by simple 1‑to‑1 sentence order?  Set to True for
# “sentence‑by‑sentence” alignment, False to keep the original
# anchor/DP algorithm.
ALIGN_BY_SENTENCE = True

# Abbreviations that should NOT trigger a sentence break
ABBREVIATIONS = frozenset({
    # English
    "e.g", "i.e", "ca", "cf", "etc", "vs", "viz", "approx", "esp",
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "No",
    "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    # Afrikaans
    "bv", "m.a.w", "d.w.s", "oa", "o.a",
    # German
    "z.B", "u.a", "bzw", "bspw",
    # French
    "Mme", "Mlle", "env",
    # Dutch
    "bijv", "enz", "m.a.w", "d.w.z", "nl", "i.o"
    # General / Music
    "Nr", "Op", "op",
})

# The separator that divides source from target in Input.txt
SEPARATOR = "==="

# Name of the input file (must be in the same directory as the script)
INPUT_FILENAME = "Input.txt"

# -----------------------------------------------------------------
#  OPTIONAL NLTK IMPORT (for a smarter sentence tokenizer)
# -----------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    nltk.data.find('tokenizers/punkt')
except Exception:          # nltk not installed or data missing
    nltk = None


# ═══════════════════════════════════════════════════════════════════
# 0. INPUT FILE PARSING
# ═══════════════════════════════════════════════════════════════════

def parse_input_file(filepath: Path) -> Tuple[str, str, str, str]:
    """
    Parse the Input.txt file.

    Expected format:
        SourceLanguageName
        [source text — one or more lines]

        ===
        TargetLanguageName
        [target text — one or more lines]

    Returns (source_language, source_text, target_language, target_text).
    """
    if not filepath.is_file():
        print(f"  ✗ Input file not found: {filepath}")
        print(f"    Please create {INPUT_FILENAME} in the same directory as this script.")
        sys.exit(1)

    content = filepath.read_text(encoding="utf-8")

    # Split on the separator line
    if SEPARATOR not in content:
        print(f"  ✗ Separator '{SEPARATOR}' not found in {filepath}")
        print(f"    The file must contain '{SEPARATOR}' on its own line between the two texts.")
        print()
        print(f"    Expected format:")
        print(f"      SourceLanguageName")
        print(f"      [source text...]")
        print(f"      ===")
        print(f"      TargetLanguageName")
        print(f"      [target text...]")
        sys.exit(1)

    parts = content.split(SEPARATOR, maxsplit=1)

    # ── Parse source section (everything before ===) ──

    src_section = parts[0].strip()
    src_lines = src_section.split("\n", maxsplit=1)

    src_lang = src_lines[0].strip()
    if not src_lang:
        print("  ✗ Source language name is missing.")
        print("    The first line of the file should be the source language name (e.g. English).")
        sys.exit(1)

    if not _is_valid_language_name(src_lang):
        print(f"  ✗ Invalid source language name: \"{src_lang}\"")
        print("    The language name must be a single word (e.g. English, Afrikaans, German).")
        print("    It looks like the first line of your file is actual text, not a language name.")
        print()
        print("    Expected format:")
        print("      English")
        print("      [source text...]")
        print("      ===")
        print("      Afrikaans")
        print("      [target text...]")
        sys.exit(1)

    src_text = src_lines[1].strip() if len(src_lines) > 1 else ""

    # ── Parse target section (everything after ===) ──

    tgt_section = parts[1].strip()
    tgt_lines = tgt_section.split("\n", maxsplit=1)

    tgt_lang = tgt_lines[0].strip()
    if not tgt_lang:
        print("  ✗ Target language name is missing.")
        print(f"    The first line after '{SEPARATOR}' should be the target language name (e.g. Afrikaans).")
        sys.exit(1)

    if not _is_valid_language_name(tgt_lang):
        print(f"  ✗ Invalid target language name: \"{tgt_lang}\"")
        print("    The language name must be a single word (e.g. English, Afrikaans, German).")
        print("    It looks like the first line after === is actual text, not a language name.")
        print()
        print("    Expected format:")
        print("      English")
        print("      [source text...]")
        print("      ===")
        print("      Afrikaans")
        print("      [target text...]")
        sys.exit(1)

    tgt_text = tgt_lines[1].strip() if len(tgt_lines) > 1 else ""

    if not src_text:
        print("  ✗ Source text is empty.")
        sys.exit(1)
    if not tgt_text:
        print("  ✗ Target text is empty.")
        sys.exit(1)

    return src_lang, src_text, tgt_lang, tgt_text


def _is_valid_language_name(name: str) -> bool:
    """
    Check whether *name* looks like a valid language name.

    A valid language name is a single word (or hyphenated word) with
    only letters, digits, hyphens, and apostrophes — no brackets,
    punctuation, or spaces. Must be between 2 and 30 characters.

    Valid:   English, Afrikaans, Middle-High-German, O'zbek
    Invalid: In Vienna's [g2]Theater..., "French Revolution", a.b.c
    """
    if len(name) < 2 or len(name) > 30:
        return False
    # Language names should be a single word — no spaces, brackets, dots, etc.
    # Allowed: letters, digits, hyphens, apostrophes
    if not re.match(r"^[\w'-]+$", name, flags=re.UNICODE):
        return False
    # Must contain at least one letter
    if not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", name):
        return False
    return True


def _sanitise_filename(name: str) -> str:
    """
    Sanitise a string so it's safe to use as a filename component.
    Removes or replaces characters that are invalid in filenames
    on Windows, macOS, and Linux.
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")
    # Remove any character that isn't alphanumeric, underscore, or hyphen
    name = re.sub(r"[^\w-]", "", name, flags=re.UNICODE)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    # Strip leading/trailing underscores
    name = name.strip("_")
    # Limit length
    name = name[:50]
    return name


# ═══════════════════════════════════════════════════════════════════
# 1. SENTENCE SPLITTING (now also on colon)
# ═══════════════════════════════════════════════════════════════════

def split_sentences(text: str) -> List[str]:
    """
    Split *text* into sentences.

    - If ``ALIGN_BY_SENTENCE`` is False we keep the original
      sophisticated splitter (which also respects tags, abbreviations,
      etc.).
    - If ``ALIGN_BY_SENTENCE`` is True we *prefer* the NLTK
      Punkt tokenizer (when available) because it gives a cleaner,
      language‑aware split.  When NLTK isn’t present we fall back to
      the original regex‑based splitter.
    - In **both** cases we additionally split on a colon (:) that is
      followed by whitespace and a capital letter or an opening tag.
    """
    # Normalise line endings – works for both paths
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ----------------------------------------------------------
    #  Simple 1‑to‑1 mode – try NLTK first
    # ----------------------------------------------------------
    if ALIGN_BY_SENTENCE:
        if nltk is not None:
            try:
                # Protect tags like [g1] temporarily so NLTK doesn’t split inside them
                placeholder = "§TAG§"
                protected = re.sub(r"\[/?[^\]]+\]", placeholder, text)
                raw_sentences = sent_tokenize(protected)
                # Restore tags
                sentences = [s.replace(placeholder, lambda m: m.group(0)) for s in raw_sentences]
                sentences = [s.strip() for s in sentences if s.strip()]
                # Post‑process colon splits
                return _split_on_colon(sentences)
            except Exception:   # pragma: no‑cover – fallback to regex version
                pass

        # No NLTK → fall back to the original splitter (which now knows about ':')
        sentences = _split_sentences_legacy(text)
        return _split_on_colon(sentences)

    # ----------------------------------------------------------
    #  Original “smart” mode (unchanged, but colon-aware)
    # ----------------------------------------------------------
    sentences = _split_sentences_legacy(text)
    return _split_on_colon(sentences)


def _split_on_colon(sentences: List[str]) -> List[str]:
    """
    Given an initial list of sentences, further split any element that
    contains a *colon* followed by whitespace and a capital letter (or an opening tag).
    """
    result: List[str] = []
    for s in sentences:
        # Split after a colon when the next token starts with A‑Z or '['
        parts = re.split(r'(?<=:)\s+(?=[A-Z\[])', s)
        for part in parts:
            part = part.strip()
            if part:
                result.append(part)
    return result


def _split_sentences_legacy(text: str) -> List[str]:
    """
    Original, more sophisticated splitter that also tries to keep
    custom markup tags intact and merges back false splits.
    """
    # ---- Normalise line endings -------------------------------------------------
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # ---- Single newlines → spaces (within paragraphs) -------------------------
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r" {2,}", " ", text)

    # ---- Protect patterns that look like sentence boundaries but aren't ----
    text = text.replace("...", "§ELL§")                     # ellipsis
    text = re.sub(r"(\d)\.(\d)", r"\1§DOT§\2", text)        # decimals
    text = re.sub(r"\b([A-Z])\.", r"\1§DOT§", text)          # initials

    for abbr in sorted(ABBREVIATIONS, key=len, reverse=True):
        pattern = re.escape(abbr) + r"\.(?=\s)"
        text = re.sub(pattern, abbr + "§DOT§", text, flags=re.IGNORECASE)

    # ---- Insert break markers after closing tags that start a new sentence ----
    text = re.sub(r"(\[/g\d+\])\s+(?=[A-Z\[])", r"\1 §BRK§ ", text)
    text = re.sub(r"(\[/g1\])\s+(?=\[)", r"\1 §BRK§ ", text)

    # ---- Primary split on punctuation (now includes colon) --------------------
    # Split after . ! ? or : when followed by whitespace and a capital letter or '['
    parts = re.split(r"(?<=[.!?:])\s+(?=[A-Z\[])", text)

    # Also split on explicit break markers inserted above
    expanded: List[str] = []
    for part in parts:
        sub_parts = re.split(r"\s*§BRK§\s*", part)
        expanded.extend(sp for sp in sub_parts if sp.strip())
    parts = expanded

    # ---- Merge false splits ----------------------------------------------------
    merged: List[str] = []
    for part in parts:
        if merged:
            prev = merged[-1]
            if (_is_abbreviation_break(prev) or
                _is_initial_break(prev) or
                _is_orphan_start(part)):
                merged[-1] += " " + part
                continue
        merged.append(part)

    # ---- Second pass: merge short orphans --------------------------------------
    merged = _merge_orphans(merged)

    # ---- Restore placeholders ---------------------------------------------------
    sentences = []
    for s in merged:
        s = s.replace("§DOT§", ".").replace("§ELL§", "...").replace("§BRK§", "")
        s = s.strip()
        if s:
            sentences.append(s)

    return sentences


def _is_abbreviation_break(prev_text: str) -> bool:
    """Check whether *prev_text* ends with an abbreviation (not a real sentence break)."""
    prev_text = prev_text.rstrip()
    prev_text_check = prev_text.replace("§DOT§", ".")
    match = re.search(r"(\w+)\.$", prev_text_check)
    if match:
        word = match.group(1)
        if word.lower() in {a.rstrip(".") for a in ABBREVIATIONS}:
            return True
    return False


def _is_initial_break(prev_text: str) -> bool:
    """Check if the previous text ends with an initial (single letter + dot)."""
    prev_text = prev_text.rstrip()
    if re.search(r"[A-Z]§DOT§$", prev_text):
        return True
    return False


def _is_orphan_start(part: str) -> bool:
    """
    Check if *part* looks like an orphan fragment that should be merged
    with the previous sentence. Examples: "This", "contrasted sharply…".
    """
    part_stripped = part.strip()
    if len(part_stripped) < 20 and not re.match(r"\[g\d+\]|\[\d+\]", part_stripped):
        return True
    return False


def _merge_orphans(sentences: List[str]) -> List[str]:
    """
    Second pass: merge very short sentences into the previous one.
    A sentence is considered an orphan if it's < 25 characters and
    doesn't look like a standalone heading/label.
    """
    if not sentences:
        return sentences

    result = [sentences[0]]
    for s in sentences[1:]:
        stripped = s.strip()
        is_label = bool(re.match(r"\[g\d+\]", stripped)) and len(stripped) < 80

        if len(stripped) < 25 and not is_label:
            result[-1] += " " + stripped
        else:
            result.append(s)

    return result


# ═══════════════════════════════════════════════════════════════════
# 2. TAG EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def extract_tags(sentence: str) -> Tuple[str, ...]:
    """
    Extract markup tags from a sentence.
    Tags are things like [g1], [g2], [1], [2], [5], [11], etc.
    Returns a sorted tuple for easy comparison.
    """
    tags = re.findall(r"\[g\d+\]|\[\d+\]", sentence)
    return tuple(sorted(tags))


# ═══════════════════════════════════════════════════════════════════
# 3. ALIGNMENT ALGORITHM
# ═══════════════════════════════════════════════════════════════════
#
# Two‑pass approach:
#   Pass 1  —  Find high‑confidence anchor pairs using tag matching
#   Pass 2  —  Fill gaps between anchors using dynamic programming
#              (supports 1:1, 1:2, 2:1 alignments)
# ═══════════════════════════════════════════════════════════════════

def find_anchors(
    src_sentences: List[str],
    tgt_sentences: List[str],
) -> List[Tuple[int, int]]:
    """
    Find anchor pairs (i, j) where src[i] and tgt[j] share markup tags.
    Returns a list of (src_idx, tgt_idx) pairs, guaranteed monotonically
    increasing in both indices (no crossings).
    """
    src_tags = [extract_tags(s) for s in src_sentences]
    tgt_tags = [extract_tags(s) for s in tgt_sentences]

    candidates: List[Tuple[int, int, int]] = []

    for i, stags in enumerate(src_tags):
        if not stags:
            continue
        for j, ttags in enumerate(tgt_tags):
            if not ttags:
                continue
            overlap = len(set(stags) & set(ttags))
            if overlap > 0:
                candidates.append((i, j, overlap))

    candidates.sort(key=lambda x: -x[2])

    anchors: List[Tuple[int, int]] = []
    used_src: Set[int] = set()
    used_tgt: Set[int] = set()

    for i, j, _score in candidates:
        if i in used_src or j in used_tgt:
            continue
        if _is_monotonic(anchors, i, j):
            anchors.append((i, j))
            used_src.add(i)
            used_tgt.add(j)

    anchors.sort(key=lambda x: x[0])
    return anchors


def _is_monotonic(
    existing: List[Tuple[int, int]], new_i: int, new_j: int
) -> bool:
    """Check that adding (new_i, new_j) doesn't cross any existing anchor."""
    for ei, ej in existing:
        if (ei < new_i and ej > new_j) or (ei > new_i and ej < new_j):
            return False
    return True


def _simple_sentence_align(src: List[str], tgt: List[str]) -> List[Tuple[str, str]]:
    """
    Pair sentences strictly by order, padding the shorter list with empty strings.
    """
    return [(s, t) for s, t in zip_longest(src, tgt, fillvalue="")]


def align_sentences(
    src_sentences: List[str],
    tgt_sentences: List[str],
) -> List[Tuple[str, str]]:
    """
    Align two lists of sentences.

    If ``ALIGN_BY_SENTENCE`` is True we skip the anchor/DP machinery
    and simply pair sentences in order.  Otherwise we fall back to the
    original, more sophisticated algorithm.
    """
    # ----------------------------------------------------------------------
    #  Simple 1‑to‑1 mode
    # ----------------------------------------------------------------------
    if ALIGN_BY_SENTENCE:
        return _simple_sentence_align(src_sentences, tgt_sentences)

    # ----------------------------------------------------------------------
    #  Original (anchor + DP) mode – unchanged
    # ----------------------------------------------------------------------
    n = len(src_sentences)
    m = len(tgt_sentences)

    if n == 0 and m == 0:
        return []
    if n == 0:
        return [("", t) for t in tgt_sentences]
    if m == 0:
        return [(s, "") for s in src_sentences]

    anchors = find_anchors(src_sentences, tgt_sentences)

    result: List[Tuple[str, str]] = []
    prev_si, prev_ti = -1, -1

    for si, ti in anchors:
        gap_src = src_sentences[prev_si + 1 : si]
        gap_tgt = tgt_sentences[prev_ti + 1 : ti]
        gap_pairs = _align_gap(gap_src, gap_tgt)
        result.extend(gap_pairs)
        result.append((src_sentences[si], tgt_sentences[ti]))
        prev_si, prev_ti = si, ti

    gap_src = src_sentences[prev_si + 1 :]
    gap_tgt = tgt_sentences[prev_ti + 1 :]
    gap_pairs = _align_gap(gap_src, gap_tgt)
    result.extend(gap_pairs)

    return result


def _align_gap(
    gap_src: List[str], gap_tgt: List[str]
) -> List[Tuple[str, str]]:
    """
    Align a gap region (between two anchors) using dynamic programming.
    Supports 1:1, 1:2 (merge in target), and 2:1 (split from source).
    """
    n = len(gap_src)
    m = len(gap_tgt)

    if n == 0 and m == 0:
        return []
    if n == 0:
        return [("", t) for t in gap_tgt]
    if m == 0:
        return [(s, "") for s in gap_src]

    if n == m:
        return list(zip(gap_src, gap_tgt))

    return _dp_align_gap(gap_src, gap_tgt)


def _dp_align_gap(
    gap_src: List[str], gap_tgt: List[str]
) -> List[Tuple[str, str]]:
    """
    DP alignment for gaps with unequal sentence counts.

    Operations:
      1:1  — src[i] ↔ tgt[j]
      skip_src — src[i] has no match
      skip_tgt — tgt[j] has no match
      1:2  — src[i] ↔ tgt[j] + tgt[j+1]
      2:1  — src[i] + src[i+1] ↔ tgt[j]
    """
    n = len(gap_src)
    m = len(gap_tgt)

    INF = float("inf")
    SKIP_PENALTY = 3.0
    MERGE_PENALTY = 0.3

    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    bt = [[-1] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue

            candidates = []

            if i > 0 and j > 0:
                cost = (
                    dp[i - 1][j - 1]
                    + _match_cost(gap_src[i - 1], gap_tgt[j - 1], i - 1, j - 1, n, m)
                )
                candidates.append((cost, 0))

            if i > 0:
                cost = dp[i - 1][j] + SKIP_PENALTY
                candidates.append((cost, 1))

            if j > 0:
                cost = dp[i][j - 1] + SKIP_PENALTY
                candidates.append((cost, 2))

            if i > 0 and j > 1:
                merged_tgt = gap_tgt[j - 2] + " " + gap_tgt[j - 1]
                cost = (
                    dp[i - 1][j - 2]
                    + _match_cost(gap_src[i - 1], merged_tgt, i - 1, j - 2, n, m)
                    + MERGE_PENALTY
                )
                candidates.append((cost, 3))

            if i > 1 and j > 0:
                merged_src = gap_src[i - 2] + " " + gap_src[i - 1]
                cost = (
                    dp[i - 2][j - 1]
                    + _match_cost(merged_src, gap_tgt[j - 1], i - 2, j - 1, n, m)
                    + MERGE_PENALTY
                )
                candidates.append((cost, 4))

            best_cost, best_op = min(candidates, key=lambda x: x[0])
            dp[i][j] = best_cost
            bt[i][j] = best_op

    pairs: List[Tuple[str, str]] = []
    i, j = n, m

    while i > 0 or j > 0:
        op = bt[i][j]

        if op == 0:
            pairs.append((gap_src[i - 1], gap_tgt[j - 1]))
            i -= 1
            j -= 1
        elif op == 1:
            pairs.append((gap_src[i - 1], ""))
            i -= 1
        elif op == 2:
            pairs.append(("", gap_tgt[j - 1]))
            j -= 1
        elif op == 3:
            pairs.append((gap_src[i - 1], gap_tgt[j - 2] + " " + gap_tgt[j - 1]))
            i -= 1
            j -= 2
        elif op == 4:
            pairs.append((gap_src[i - 2] + " " + gap_src[i - 1], gap_tgt[j - 1]))
            i -= 2
            j -= 1
        else:
            break

    pairs.reverse()
    return pairs


def _match_cost(
    src: str,
    tgt: str,
    src_idx: int,
    tgt_idx: int,
    src_len: int,
    tgt_len: int,
) -> float:
    """
    Cost of aligning *src* with *tgt*.
    Lower cost = better match.

    Signals:
      - Tag overlap (weight 3.0)
      - Position similarity (weight 1.5)
      - Length ratio (weight 0.5)
    """
    src_tags = set(extract_tags(src))
    tgt_tags = set(extract_tags(tgt))

    if src_tags and tgt_tags:
        overlap = len(src_tags & tgt_tags)
        union = len(src_tags | tgt_tags)
        tag_sim = overlap / union
    elif not src_tags and not tgt_tags:
        tag_sim = 0.5
    else:
        tag_sim = 0.0

    src_pos = src_idx / max(src_len - 1, 1)
    tgt_pos = tgt_idx / max(tgt_len - 1, 1)
    pos_sim = 1.0 - abs(src_pos - tgt_pos)

    sl, tl = len(src), len(tgt)
    if sl > 0 and tl > 0:
        len_sim = min(sl, tl) / max(sl, tl)
    else:
        len_sim = 0.0

    cost = -(3.0 * tag_sim + 1.5 * pos_sim + 0.5 * len_sim)
    return cost


# ═══════════════════════════════════════════════════════════════════
# 4. OUTPUT GENERATION
# ═══════════════════════════════════════════════════════════════════

def write_csv(
    pairs: List[Tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    filepath: Path,
) -> None:
    """Write aligned pairs to a CSV file."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([src_lang, tgt_lang])
        for src, tgt in pairs:
            writer.writerow([src, tgt])
    print(f"  ✓ CSV saved to {filepath}")


def write_json(
    pairs: List[Tuple[str, str]],
    src_lang: str,
    tgt_lang: str,
    filepath: Path,
) -> None:
    """Write aligned pairs to a JSON file."""
    src_key = src_lang.lower().replace(" ", "_")
    tgt_key = tgt_lang.lower().replace(" ", "_")

    data = [{src_key: src, tgt_key: tgt} for src, tgt in pairs]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ JSON saved to {filepath}")


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    # Determine the directory where this script lives
    script_dir = Path(__file__).resolve().parent
    input_path = script_dir / INPUT_FILENAME
    output_dir = script_dir

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║       Bilingual Sentence Aligner                    ║")
    print("║  Reads from Input.txt → outputs CSV + JSON          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Parse input file ──

    src_lang, src_text, tgt_lang, tgt_text = parse_input_file(input_path)
    print(f"  Source language : {src_lang}")
    print(f"  Target language : {tgt_lang}")
    print(f"  Source text     : {len(src_text)} characters")
    print(f"  Target text     : {len(tgt_text)} characters")
    print()

    # ── Sentence splitting ──

    src_sentences = split_sentences(src_text)
    tgt_sentences = split_sentences(tgt_text)

    print(f"  {src_lang}: {len(src_sentences)} sentences")
    print(f"  {tgt_lang}: {len(tgt_sentences)} sentences")

    # ── Alignment ──

    pairs = align_sentences(src_sentences, tgt_sentences)

    print(f"  Aligned into {len(pairs)} pairs")
    print()

    # ── Preview ──

    print("  ─── Preview (first 5 pairs) ───")
    for idx, (s, t) in enumerate(pairs[:5], 1):
        s_preview = s[:70] + ("\u2026" if len(s) > 70 else "")
        t_preview = t[:70] + ("\u2026" if len(t) > 70 else "")
        print(f"  {idx}. {src_lang}: {s_preview}")
        print(f"     {tgt_lang}: {t_preview}")
        print()

    if len(pairs) > 5:
        print(f"  \u2026 and {len(pairs) - 5} more pairs")
        print()

    # ── Write outputs ──

    out_base = f"aligned_{_sanitise_filename(src_lang.lower())}_{_sanitise_filename(tgt_lang.lower())}"
    csv_path = output_dir / f"{out_base}.csv"
    json_path = output_dir / f"{out_base}.json"

    write_csv(pairs, src_lang, tgt_lang, csv_path)
    write_json(pairs, src_lang, tgt_lang, json_path)

    print()
    print("  Done! \u2713")


if __name__ == "__main__":
    main()