# Changelog

## 2026-11-22 — Atomic collection admission

- Added collection-wide document, page and extracted-character limits.
- Validate an upload batch before chunking or embedding starts.
- Preserve the prior document set and index if admission or rebuilding fails.

## 2026-11-22 — Inspectable grounding evaluations

- Added stable case identifiers and per-case retrieval, answer and policy results.
- Added optional JSON report output to the end-to-end evaluation runner.
- Retain each hosted grounding report as a 14-day GitHub Actions artifact.

## 2026-11-22 — Bounded document ingestion

- Added post-parse page and extracted-character limits before chunking and embedding.
- Added explicit empty-text/OCR guidance and deterministic boundary tests.
- Documented that content limits do not sandbox the underlying PDF parser.

## 2026-11-22 — End-to-end answer evaluation

- Added MiniLM retrieval-to-answer evaluation across six answerable and two unanswerable cases.
- Measure expected evidence-term accuracy, abstention accuracy and citation validity separately.
- Added a hosted CI regression floor for every pipeline metric.
- Corrected citation auditing so decimal points are not treated as sentence boundaries.

## 2026-11-22 — Credential-free extractive Q&A

- Added a deterministic extractive answer mode that returns retrieved sentences with citations.
- Added explicit abstention when retrieved evidence has no meaningful term overlap.
- Allowed PDF indexing and extractive Q&A without a Groq API key.
- Disabled only generative controls when provider credentials are absent.

## 2026-11-21 — Production retriever evaluation

- Added executable evaluation of the actual `all-MiniLM-L6-v2` retriever.
- Added confounding corpus passages to make the six labelled queries less trivial.
- Reused the same hit-rate-at-3 and MRR calculations across lexical and embedding rankings.
- Added the MiniLM evaluation to hosted CI with an explicit regression floor.

## 2026-11-21 — Grounding evaluation baseline

- Enforced citations on every factual answer sentence, not merely one citation somewhere in an answer.
- Added a deterministic BM25 retrieval baseline with labelled corpus/query fixtures.
- Added hit-rate-at-3 and mean-reciprocal-rank reporting plus CI regression thresholds.
- Documented that the synthetic lexical baseline does not validate MiniLM or model answer quality.

## 2026-11-21

- Added stable evidence labels and post-generation citation auditing for Q&A.
- Marked retrieved document text as untrusted to reduce prompt-injection risk.
- Changed public deployments to bring-your-own-key by default; server-owned
  Groq keys now require an explicit private-deployment opt-in.
- Escaped document names and model-generated values before custom HTML output.
- Added a 10 MB per-document upload limit and offline grounding-policy tests.
- Replaced syntax-only CI with linting, tests and targeted compilation.

