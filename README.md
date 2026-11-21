# Document Intelligence Platform

A multi-document analysis application that allows you to upload several PDF files at once and extract structured knowledge from them collectively. The platform generates per-document summaries, identifies named entities, compares documents against each other, and answers natural-language questions with audited evidence labels through a web interface.
---
HOSTED PREVIEW (Streamlit account access currently required): https://document-intelligence-platform-swnwnjuqwbc96kudr5ekmr.streamlit.app/

---

## What it does

Most document analysis tools process one file at a time. This platform treats an entire collection of PDFs as a unified knowledge base. Once you upload your files, every feature operates across all of them simultaneously:

- Summarisation produces a 4-5 sentence factual summary for each document independently.
- Entity extraction identifies organisations, dates, monetary figures, key topics, and locations within each document.
- Cross-document comparison synthesises common themes, key differences, relationships between documents, and strategic takeaways.
- Unified Q&A asks the model to cite stable retrieved-evidence labels and audits
  every factual sentence and audits the returned labels before presenting the answer as grounded. Failed audits
  are shown as unverified rather than silently treated as valid attribution.

---

## Architecture

PDF files are parsed page by page using PyPDF. Each page is split into overlapping 500-character chunks using LangChain's RecursiveCharacterTextSplitter. All chunks from all documents are embedded with the HuggingFace `all-MiniLM-L6-v2` sentence-transformer model and indexed together in a single FAISS vector store. Every query to the Q&A system retrieves the five most semantically similar chunks from that shared index, regardless of which file they came from. Retrieved chunks are wrapped in explicit untrusted-data boundaries and assigned stable labels such as `[S1]`; the model is instructed to ignore instructions embedded in documents and cite those labels. A deterministic post-generation audit rejects unknown labels and flags uncited answers.

Summarisation, entity extraction, comparison, and question answering use the
Groq-hosted `llama-3.1-8b-instant` model via the LangChain Groq integration.
Provider accounts, quotas and terms remain the user's responsibility.

---

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM inference | Groq API (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector store | FAISS (CPU) |
| PDF parsing | PyPDF |
| Orchestration | LangChain RetrievalQA |

---

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/LAKSHAY-ATREJA/document-intelligence-platform
cd document-intelligence-platform

python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

The first time you run the application, the sentence-transformer model (~90 MB) is downloaded automatically from HuggingFace and cached locally.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Private deployments only | Optional server-owned Groq key |
| `ALLOW_SERVER_API_KEY` | No | Must be explicitly `true` before the server-owned key is used |

Public deployments require each user to enter their own Groq key in the password
field. A server-owned key on an anonymous app can be consumed by any visitor, so
it is ignored unless `ALLOW_SERVER_API_KEY=true` is explicitly set for a
controlled/private deployment. The application does not write user-entered keys
to disk, but Streamlit session memory and the Groq provider still process them.

To obtain a free Groq API key, create an account at https://console.groq.com, navigate to API Keys, and generate a new key. The free tier allows several hundred requests per day at no cost.

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
# Edit .env and set GROQ_API_KEY=gsk_your_key_here
```

---

## Running locally

```bash
streamlit run app.py
```

Streamlit opens a browser tab at `http://localhost:8501`. Enter your Groq API key in the sidebar, upload one or more PDF files, and use the four tabs — Summaries, Entity Extraction, Cross-Doc Analysis, and Q&A — to analyse your documents.

---

## Running the demo script

`demo.py` runs the three core analytical functions (summarisation, entity extraction, cross-document comparison) against two short sample texts that are embedded directly in the script. No PDF upload is needed. Use this to confirm that your installation and API key are working before uploading real documents.

```bash
python3 demo.py
```

If `GROQ_API_KEY` is not set in your environment, the script will prompt you to enter it interactively.

Sample output excerpt:

```
======================================================================
  STEP 1: Document Summarisation
======================================================================

  Summarising: quarterly_report_q1.txt ...

  Summary:
  Acme Corporation reported strong Q1 2024 results with total revenue of
  $4.8 billion, a 12% year-over-year increase. The Cloud Services division
  led growth at 34%, contributing $1.9 billion. The company completed the
  acquisition of DataSync Inc. for $340 million and reaffirmed full-year
  revenue guidance of $19.5–20.0 billion.

======================================================================
  STEP 3: Cross-Document Comparison
======================================================================

  ## Comparison Analysis

  ### Common Themes
  Both documents focus on cloud infrastructure growth ...
```

---

## Deployment

### Streamlit Community Cloud (recommended, free)

1. Fork this repository to your GitHub account.
2. Visit https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select your fork, set the main file to `app.py`, and click "Deploy".
4. Leave server secrets unset for a public deployment; users supply their own Groq key.

The hosted preview linked above currently requires Streamlit account access, so
it is not claimed as a public anonymous demo. The application is configured for
bring-your-own-key mode; it is not an offline model and does not include a
shared inference key.

### Render (free tier)

The repository includes `render.yaml` with the correct build and start commands. To deploy:

1. Create a free account at https://render.com.
2. Click "New Web Service" and connect your GitHub repository.
3. Render detects `render.yaml` automatically. Click "Create Web Service".
4. For a private deployment only, add `GROQ_API_KEY` and set
   `ALLOW_SERVER_API_KEY=true`. Keep the default bring-your-own-key mode when
   the service is publicly reachable.

The free Render tier spins down after 15 minutes of inactivity; the first request after that may take 30-60 seconds to respond.

### Any platform with Docker or a Procfile

The `Procfile` contains:

```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true
```

This is compatible with Heroku, Railway, Fly.io, and any platform that respects the `$PORT` convention.

---

## Use cases

- Legal and compliance teams reviewing contracts, policies, or regulations across multiple versions or jurisdictions.
- Researchers comparing papers, reports, or datasets from different sources.
- Business analysts extracting figures and entities from earnings reports or market research.
- Due diligence workflows where multiple vendor documents need to be compared quickly.
- Internal knowledge management when important information is spread across many uploaded PDFs.

---

## Notes on PDF compatibility

The parser works on text-based PDFs where text is stored as selectable characters. Scanned PDFs (images of pages) produce no extractable text and will result in empty summaries. If you need to process scanned documents, run them through an OCR tool such as `ocrmypdf` before uploading.

Each uploaded PDF is limited to 10 MB. Retrieval is evaluated only through
deterministic grounding-policy tests in CI at present; no claim is made that the
MiniLM embeddings or Groq model have been benchmarked on a representative legal,
financial, or safety-critical corpus. Citation presence proves that a retrieved
chunk was referenced, not that the generated interpretation is correct.

## Quality checks

```bash
pip install -r requirements-dev.txt
ruff check app.py rag_core.py tests
pytest -q
```

The offline tests exercise evidence labelling, prompt-injection boundaries,
unknown-citation rejection, missing-citation detection and explicit abstention.
They do not call Groq or download an embedding model.

`python evaluation/retrieval_eval.py --minimum-hit-rate 0.8` runs a checked-in
BM25 baseline over six labelled, synthetic retrieval cases and reports hit
rate at 3 and mean reciprocal rank. It is a deterministic regression floor for
the evaluation plumbing, not evidence that the application's MiniLM embeddings
generalise to real legal, financial or safety-critical documents. A larger,
representative embedding benchmark remains required before making such a claim.

---

Built by Lakshay Atreja — https://github.com/LAKSHAY-ATREJA

## Licence

MIT — see [LICENSE](LICENSE).


