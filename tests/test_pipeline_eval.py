from evaluation.pipeline_eval import evaluate_cases, evaluate_pipeline


def test_pipeline_metrics_separate_answers_abstentions_and_citations():
    corpus = [
        {"id": "retention", "text": "Records are retained for seven years."},
        {"id": "revenue", "text": "Revenue increased this year."},
    ]
    cases = [
        {
            "slice": "answerable",
            "question": "How long are records retained?",
            "expected_terms": ["seven years"],
            "should_abstain": False,
        },
        {
            "slice": "out_of_scope",
            "question": "What is the vacation allowance?",
            "expected_terms": [],
            "should_abstain": True,
        },
    ]
    metrics = evaluate_cases(corpus, cases, [["retention", "revenue"], ["revenue"]])
    assert metrics["cases"] == 2
    assert metrics["answerable_accuracy"] == 1.0
    assert metrics["abstention_accuracy"] == 1.0
    assert metrics["citation_pass_rate"] == 1.0
    assert metrics["slices"] == {
        "answerable": {"cases": 1, "accuracy": 1.0, "citation_pass_rate": 1.0},
        "out_of_scope": {"cases": 1, "accuracy": 1.0, "citation_pass_rate": 1.0},
    }


def test_pipeline_report_exposes_per_case_evidence_and_failures():
    corpus = [{"id": "retention", "text": "Records are retained for seven years."}]
    cases = [
        {
            "id": "retention-period",
            "slice": "direct_lookup",
            "question": "How long are records retained?",
            "expected_terms": ["eight years"],
            "should_abstain": False,
        },
        {
            "id": "unknown-location",
            "slice": "out_of_scope",
            "question": "Where is the office?",
            "expected_terms": [],
            "should_abstain": True,
        },
    ]

    report = evaluate_pipeline(corpus, cases, [["retention"], ["retention"]])

    assert report["evaluation"] == {"retriever": "all-MiniLM-L6-v2", "top_k": 5}
    assert report["case_results"][0] == {
        "id": "retention-period",
        "slice": "direct_lookup",
        "question": "How long are records retained?",
        "should_abstain": False,
        "retrieved_ids": ["retention"],
        "answer": "Records are retained for seven years. [S1]",
        "expected_terms_passed": False,
        "abstention_passed": False,
        "citation_passed": True,
    }
    assert report["metrics"]["answerable_accuracy"] == 0.0
    assert report["metrics"]["slices"]["direct_lookup"]["accuracy"] == 0.0
