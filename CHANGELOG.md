# Changelog

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

