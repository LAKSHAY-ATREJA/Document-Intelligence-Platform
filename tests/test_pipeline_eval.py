from evaluation.pipeline_eval import evaluate_cases


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
