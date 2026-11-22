from evaluation.pipeline_eval import evaluate_cases, evaluate_pipeline


def test_pipeline_metrics_separate_answers_abstentions_and_citations():
    corpus = [
        {"id": "retention", "text": "Records are retained for seven years."},
        {"id": "revenue", "text": "Revenue increased this year."},
    ]
    cases = [
        {
            "question": "How long are records retained?",
            "expected_terms": ["seven years"],
            "should_abstain": False,
        },
        {
            "question": "What is the vacation allowance?",
            "expected_terms": [],
            "should_abstain": True,
        },
    ]
    metrics = evaluate_cases(corpus, cases, [["retention", "revenue"], ["revenue"]])
    assert metrics == {
        "cases": 2,
        "answerable_accuracy": 1.0,
        "abstention_accuracy": 1.0,
        "citation_pass_rate": 1.0,
    }


def test_pipeline_report_exposes_per_case_evidence_and_failures():
    corpus = [{"id": "retention", "text": "Records are retained for seven years."}]
    cases = [
        {
            "id": "retention-period",
            "question": "How long are records retained?",
            "expected_terms": ["eight years"],
            "should_abstain": False,
        },
        {
            "id": "unknown-location",
            "question": "Where is the office?",
            "expected_terms": [],
            "should_abstain": True,
        },
    ]

    report = evaluate_pipeline(corpus, cases, [["retention"], ["retention"]])

    assert report["evaluation"] == {"retriever": "all-MiniLM-L6-v2", "top_k": 5}
    assert report["case_results"][0] == {
        "id": "retention-period",
        "question": "How long are records retained?",
        "should_abstain": False,
        "retrieved_ids": ["retention"],
        "answer": "Records are retained for seven years. [S1]",
        "expected_terms_passed": False,
        "abstention_passed": False,
        "citation_passed": True,
    }
    assert report["metrics"]["answerable_accuracy"] == 0.0
