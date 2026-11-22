"""Evaluate the application's MiniLM embedding retriever on labelled fixtures."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.retrieval_eval import evaluate_rankings


def rank_with_minilm(corpus: list[dict], questions: list[str]) -> list[list[str]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_vectors = model.encode(
        [item["text"] for item in corpus], normalize_embeddings=True
    )
    query_vectors = model.encode(questions, normalize_embeddings=True)
    rankings: list[list[str]] = []
    for query_vector in query_vectors:
        scores = corpus_vectors @ query_vector
        order = sorted(
            range(len(corpus)),
            key=lambda index: (-scores[index], corpus[index]["id"]),
        )
        rankings.append([corpus[index]["id"] for index in order])
    return rankings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-hit-rate", type=float, default=0.0)
    args = parser.parse_args()
    root = Path(__file__).parent
    corpus = json.loads((root / "retrieval_corpus.json").read_text())
    queries = json.loads((root / "retrieval_queries.json").read_text())
    rankings = rank_with_minilm(corpus, [case["question"] for case in queries])
    metrics = evaluate_rankings(rankings, queries)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["hit_rate_at_3"] >= args.minimum_hit_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
