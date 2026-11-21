import json
from pathlib import Path

from evaluation.retrieval_eval import evaluate, rank_bm25


def test_bm25_ranks_direct_evidence_first():
    corpus = [
        {"id": "retention", "text": "Records are retained for seven years."},
        {"id": "insurance", "text": "Insurance coverage is five million dollars."},
    ]
    assert rank_bm25("How long are records retained?", corpus)[0] == "retention"


def test_checked_in_baseline_meets_regression_floor():
    root = Path(__file__).parents[1] / "evaluation"
    corpus = json.loads((root / "retrieval_corpus.json").read_text())
    queries = json.loads((root / "retrieval_queries.json").read_text())
    metrics = evaluate(corpus, queries)
    assert metrics["hit_rate_at_3"] >= 0.8
    assert metrics["mean_reciprocal_rank"] >= 0.8
