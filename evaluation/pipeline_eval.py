"""End-to-end MiniLM retrieval and extractive-answer regression evaluation."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.embedding_eval import rank_with_minilm
from rag_core import (
    audit_citations,
    build_evidence_context,
    extractive_answer,
)


@dataclass(frozen=True)
class EvaluationDocument:
    page_content: str
    metadata: dict


def evaluate_cases(corpus: list[dict], cases: list[dict], rankings: list[list[str]]) -> dict:
    by_id = {item["id"]: item for item in corpus}
    answerable_total = 0
    answerable_passed = 0
    abstention_total = 0
    abstention_passed = 0
    citation_passed = 0
    for case, ranking in zip(cases, rankings, strict=True):
        documents = [
            EvaluationDocument(
                page_content=by_id[document_id]["text"],
                metadata={"source_file": f"{document_id}.txt"},
            )
            for document_id in ranking[:5]
        ]
        answer = extractive_answer(case["question"], documents)
        audit = audit_citations(answer, build_evidence_context(documents))
        citation_passed += audit.passed
        if case["should_abstain"]:
            abstention_total += 1
            abstention_passed += answer == "INSUFFICIENT_EVIDENCE"
        else:
            answerable_total += 1
            lowered = answer.lower()
            answerable_passed += audit.passed and all(
                term.lower() in lowered for term in case["expected_terms"]
            )
    return {
        "cases": len(cases),
        "answerable_accuracy": answerable_passed / answerable_total,
        "abstention_accuracy": abstention_passed / abstention_total,
        "citation_pass_rate": citation_passed / len(cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-accuracy", type=float, default=0.0)
    args = parser.parse_args()
    root = Path(__file__).parent
    corpus = json.loads((root / "retrieval_corpus.json").read_text())
    cases = json.loads((root / "pipeline_cases.json").read_text())
    rankings = rank_with_minilm(corpus, [case["question"] for case in cases])
    metrics = evaluate_cases(corpus, cases, rankings)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    accuracy_passed = all(
        metrics[name] >= args.minimum_accuracy
        for name in ("answerable_accuracy", "abstention_accuracy")
    )
    return 0 if accuracy_passed and metrics["citation_pass_rate"] == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
