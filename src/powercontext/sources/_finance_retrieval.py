# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline lexical retrieval for financial-report evidence pages.

``search_pdf_pages`` receives a natural-language question and must return the
physical pages that carry the answer. The evaluation is fully offline (no model
calls) and must not read the ground-truth answers, so this module implements a
self-contained BM25 ranker over the scoped Memory corpus: it cleans the question
of instruction boilerplate, stems tokens, expands a small dictionary of
finance-domain synonyms and abbreviations, and ranks every page by BM25 with
IDF weighting.

The scorer is intentionally self-contained (a compact Porter stemmer and a
hand-written finance lexicon) so it needs no extra dependencies beyond the
standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z]+|\d+")

# Instruction sentences that add no retrieval signal and only inflate the
# query with function words. Dropping them keeps the term set focused on the
# financial content of the question.
_INSTRUCTION_SENTENCE_RE = re.compile(
    r"("
    r"respond to the question|answer in|answer the question|give a response|give a solution|"
    r"provide a response|address the question|approach the question|"
    r"base your judgment|basing your judgment|basing your judgements|basing your judgments|"
    r"assuming the perspective|who can only|rely on the details|shown within|"
    r"assume that you are|as if you are|adopting the perspective|drawing conclusions|"
    r"calculate what was asked|utilizing the line items|utilizing the information|utilize the information|"
    r"compute or extract|extract the answer|calculate \(or extract\)|calculate or extract|"
    r"have lost internet|do not have access|state that and explain|then please state"
    r")",
    re.IGNORECASE,
)

# Terms that never help locate a page. Query and page tokens are both stemmed,
# so these are the stemmed forms where relevant.
_STOPWORDS = frozenset(
    """a an the and or of to in on for with at by from as is are was were be been being has have had do does did
    this that these those it its what which who whom when how why not no nor so if then else than too very only
    own same just but out about into over after before under again further once here there all any both each few
    more most other some such s t d ll re ve m don can could would should will may might must shall
    give response respond question answer assume assuming perspective investment analyst use used using only
    details shown within statement statements rely relying relied calculate define defined outline outlined
    state amount quantity reported report directly explicitly ongoing materially primarily unadjusted less margin
    as the a an of in on for with by and or""".split()
)

# Single-word boilerplate dropped from the query only (not from pages).
_QUERY_DROP_WORDS = frozenset(
    """give response respond question answer assume assuming perspective investment analyst use used using only
    can details shown within statement statements rely relying relied calculate define defined outline outlined
    state amount quantity reported report directly explicitly ongoing materially primarily unadjusted less margin
    above below accordingly additionally also although always among annually approximately as at because before
    between both but by during each either especially etc example excluding finally following from further
    generally however if include including indeed instead least let likely mainly moreover namely neither
    nevertheless next notably often otherwise overall particularly perhaps please rather respective respectively
    second secondarily since specifically such than that then thereby therefore therefor thus typically unless
    until using various versus via whereas whether while within would yet""".split()
)

# Finance-domain synonym/abbreviation expansion. Keys are raw or stemmed query
# tokens; values are extra tokens (stemmed at query time) that the financial
# report is likely to use instead. Weights favour the original question terms
# over expanded ones, so the expansions refine rather than dominate ranking.
_SYNONYM_WEIGHT = 0.6
_FINANCE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "capex": ("capital", "expenditure", "expenditures", "spending"),
    "ebitda": ("operating", "income", "depreciation", "amortization", "profit"),
    "geograph": ("segment", "segments", "division", "divisions", "region", "regions", "country", "countries", "territories"),
    "geographies": ("segment", "segments", "division", "divisions", "region", "regions"),
    "legal": ("proceedings", "litigation", "claims", "regulatory"),
    "battl": ("proceedings", "litigation", "claims", "regulatory", "legal"),
    "ppe": ("property", "plant", "equipment"),
    "ppne": ("property", "plant", "equipment"),
    "fixed": ("property", "plant", "equipment"),
    "revenue": ("sales", "net"),
    "cogs": ("cost", "sales"),
    "dividend": ("dividends", "paid", "payout"),
    "repurchase": ("buyback", "buybacks", "repurchases"),
    "buyback": ("repurchase", "repurchases"),
    "acquisition": ("acquisitions", "acquired", "business", "combinations"),
    "receiv": ("accounts", "receivable"),
    "payabl": ("accounts", "payable"),
    "ar": ("accounts", "receivable"),
    "ap": ("accounts", "payable"),
    "margin": ("profit", "income"),
    "tax": ("taxes", "provision"),
    "restructur": ("restructuring", "impairment", "charges"),
    "impair": ("restructuring", "impairment"),
    "capital": ("expenditure", "expenditures", "spending"),
    "income": ("profit", "earnings"),
    "operat": ("profit", "earning"),
    "quick": ("current", "assets", "liabilities"),
    "liquidity": ("current", "assets", "liabilities", "cash"),
    "working": ("current", "assets", "liabilities"),
    "fcf": ("cash", "operating", "capital", "spending"),
    "roa": ("net", "income", "assets"),
    "roe": ("net", "income", "equity"),
    "d&a": ("depreciation", "amortization"),
    "inventory": ("inventories", "stock"),
    "shareholders": ("stockholders", "shareholders"),
}


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _porter_stem(word: str) -> str:
    """A compact Porter-style stemmer, sufficient for financial-report text."""

    word = word.lower()
    if len(word) <= 2:
        return word
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]
    if word.endswith("eed"):
        word = word[:-1]
    elif word.endswith("ed") or word.endswith("ing"):
        stem = word[:-3] if word.endswith("ing") else word[:-2]
        if any(vowel in stem for vowel in "aeiou"):
            word = stem
    if word.endswith("y") and len(word) > 2 and word[-2] not in "aeiou":
        word = word[:-1] + "i"
    for suffix in (
        "ational", "tional", "enci", "anci", "izer", "bli", "alli", "entli", "eli", "ousli",
        "ization", "ation", "ator", "alism", "iveness", "fulness", "ousness", "aliti", "iviti",
        "biliti", "logi",
    ):
        if word.endswith(suffix):
            if suffix in ("ation", "ator"):
                word = word[: -len(suffix)] + "e"
            elif suffix in ("alism", "aliti"):
                word = word[: -len(suffix)] + "al"
            elif suffix in ("iveness", "fulness", "ousness"):
                word = word[: -len(suffix)] + "i"
            elif suffix == "iviti":
                word = word[: -len(suffix)] + "ive"
            elif suffix == "biliti":
                word = word[: -len(suffix)] + "ble"
            elif suffix == "logi":
                word = word[: -len(suffix)] + "log"
            elif suffix == "ization":
                word = word[: -len(suffix)] + "ize"
            elif suffix == "izer":
                word = word[: -len(suffix)] + "ize"
            elif suffix == "ational":
                word = word[: -len(suffix)] + "ate"
            elif suffix == "tional":
                word = word[: -len(suffix)] + "tion"
            elif suffix == "enci":
                word = word[: -len(suffix)] + "ence"
            elif suffix == "anci":
                word = word[: -len(suffix)] + "ance"
            elif suffix == "alli":
                word = word[: -len(suffix)] + "al"
            elif suffix == "entli":
                word = word[: -len(suffix)] + "ent"
            elif suffix == "eli":
                word = word[: -len(suffix)] + "e"
            elif suffix == "ousli":
                word = word[: -len(suffix)] + "ous"
            elif suffix == "bli":
                word = word[: -len(suffix)] + "ble"
            break
    return word


def _page_body(text: str) -> str:
    """Strip the leading ``DOCUMENT_ID`` line from a page's evidence text."""

    if "\n" in text:
        return text.partition("\n")[2]
    return text


def _expand_query(question: str) -> dict[str, float]:
    """Return a mapping of stemmed query term to its BM25 weight."""

    # Drop instruction sentences and keep the content-bearing ones.
    sentences = re.split(r"(?<=[.!?])\s+", question)
    kept = [sentence for sentence in sentences if not _INSTRUCTION_SENTENCE_RE.search(sentence.lower().strip())]
    query = " ".join(kept) if kept else question

    weights: dict[str, float] = {}
    for token in _tokenize(query):
        if token in _QUERY_DROP_WORDS or (len(token) == 1 and not token.isdigit()):
            continue
        stemmed = _porter_stem(token)
        if stemmed in _STOPWORDS or len(stemmed) <= 1:
            continue
        weights[stemmed] = weights.get(stemmed, 0.0) + 1.0

    expanded: dict[str, float] = {}
    for stemmed in list(weights):
        for key, synonyms in _FINANCE_SYNONYMS.items():
            if stemmed == key or stemmed.startswith(key) or key.startswith(stemmed):
                for synonym in synonyms:
                    syn = _porter_stem(synonym)
                    if syn and syn not in weights and syn != stemmed:
                        expanded[syn] = expanded.get(syn, 0.0) + _SYNONYM_WEIGHT
    for stemmed, weight in expanded.items():
        weights[stemmed] = weights.get(stemmed, 0.0) + weight
    return weights


def rank_pages(records: Iterable[object], question: str, /) -> list[tuple[float, object]]:
    """Rank Memory entry records by BM25 relevance to ``question``.

    Returns a list of ``(score, record)`` pairs sorted by descending score, with
    ties broken by the entry id for a stable, reproducible order.
    """

    records = list(records)
    if not records:
        return []

    query_terms = _expand_query(question)
    if not query_terms:
        return [(0.0, record) for record in records]

    bodies = [_page_body(record.entry.text) for record in records]
    stemmed_pages = [[_porter_stem(token) for token in _tokenize(body)] for body in bodies]

    document_count = len(stemmed_pages)
    document_frequency: Counter[str] = Counter()
    for page in stemmed_pages:
        for token in set(page):
            document_frequency[token] += 1

    average_length = sum(len(page) for page in stemmed_pages) / document_count
    idf = {
        token: math.log(1.0 + (document_count + 0.5) / (document_frequency[token] + 0.5))
        for token in document_frequency
    }

    scored: list[tuple[float, object]] = []
    for record, page in zip(records, stemmed_pages):
        term_frequency = Counter(page)
        page_length = len(page)
        score = 0.0
        for term, weight in query_terms.items():
            count = term_frequency.get(term)
            if not count:
                continue
            denominator = count + 1.2 * (0.25 + 0.75 * page_length / average_length)
            score += weight * idf[term] * (count * 2.2 / denominator)
        scored.append((score, record))

    scored.sort(key=lambda pair: (-pair[0], getattr(pair[1].entry, "entry_id", "")))
    return scored
