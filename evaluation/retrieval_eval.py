"""Deterministic BM25 retrieval baseline and ranking metrics.

This is deliberately a lexical regression floor, not an evaluation of the
MiniLM embedding model used by the application.
"""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def rank_bm25(query: str, corpus: list[dict], *, k1: float = 1.2, b: float = 0.75) -> list[str]:
    tokenized = [tokenize(item["text"]) for item in corpus]
    average_length = sum(map(len, tokenized)) / len(tokenized)
    document_frequency = Counter(
        token for document in tokenized for token in set(document)
    )
    query_tokens = tokenize(query)
    scored: list[tuple[float, str]] = []
    for item, document in zip(corpus, tokenized, strict=True):
        frequencies = Counter(document)
        score = 0.0
        for token in query_tokens:
            frequency = frequencies[token]
            if not frequency:
                continue
            df = document_frequency[token]
            inverse_frequency = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1 - b + b * len(document) / average_length
            )
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scored.append((score, item["id"]))
    return [document_id for _, document_id in sorted(scored, key=lambda row: (-row[0], row[1]))]


def evaluate(corpus: list[dict], queries: list[dict], *, k: int = 3) -> dict[str, float | int]:
    hits = 0
    reciprocal_ranks = 0.0
    for case in queries:
        ranking = rank_bm25(case["question"], corpus)
        relevant = set(case["relevant_ids"])
        hits += bool(relevant.intersection(ranking[:k]))
        first_rank = next(
            (index for index, document_id in enumerate(ranking, start=1) if document_id in relevant),
            None,
        )
        reciprocal_ranks += 1 / first_rank if first_rank else 0
    return {
        "queries": len(queries),
        f"hit_rate_at_{k}": hits / len(queries),
        "mean_reciprocal_rank": reciprocal_ranks / len(queries),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-hit-rate", type=float, default=0.0)
    args = parser.parse_args()
    root = Path(__file__).parent
    corpus = json.loads((root / "retrieval_corpus.json").read_text())
    queries = json.loads((root / "retrieval_queries.json").read_text())
    metrics = evaluate(corpus, queries)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["hit_rate_at_3"] >= args.minimum_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
