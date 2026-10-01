# PDX Finder Agent

**An agentic assistant that recommends patient-derived xenograft (PDX) models from CancerModels.org in response to natural-language requests, and justifies every recommendation with the underlying data.**

> *"I need breast cancer PDX models that showed a complete response to paclitaxel."*
> → The agent looks up how the drug is written in the catalogue, filters 5,997 PDX models by cancer type, drug and response, and returns the 3 matching models. For each one it quotes the field values that support each criterion and gives a link to the model's record.

This is the third project of a series:

| Project | Question |
|---|---|
| [LLM Bench Oncology](#) | How well do LLMs answer biomedical questions, with and without the source paper? |
| [OncoRAG](#) | Can a retrieval system find the right paper among 10,000? |
| **PDX Finder Agent** | Can an LLM agent search a heterogeneous biobank catalogue and recommend models transparently? |

## Data: 5,997 PDX models from CancerModels.org

- **Source:** the [CancerModels.org collection on EMBL-EBI BioStudies](https://www.ebi.ac.uk/biostudies/CancerModels.Org/studies) (CC0). It has 10,348 models from more than 50 providers, including NCI PDMR, JAX, Institut Curie, St. Jude and the Novartis PDX Encyclopedia (`NIBRX-*`). The CancerModels.org REST API was returning HTTP 500 at the time, so metadata was fetched from BioStudies instead (`fetch_metadata.py`, cached and resumable).
- **Heterogeneous data unified into one schema** (`build_dataset.py`, validated with Pydantic):
  - **Clinical:** patient sex, age, ethnicity, paediatric flag.
  - **Pathological:** histology, cancer system, primary and collection site, tumour type, stage, grade, breast biomarkers.
  - **Experimental:** host strain, engraftment site and type, quality-control technique.
  - **Treatment:** drug responses measured *in the PDX* (603 models), and treatments *the patient* received (781 models).
  - **Molecular:** available data types, plus TMB and MSI where reported.
- Repeated sub-records, such as several drugs tested on one model, are kept as lists. Every model also gets a generated text description, which is used for semantic search.

## Architecture

```
User request (any language)
      │
      ▼
gpt-oss-120b (Cerebras, OpenAI-compatible)  ── ReAct loop via smolagents ToolCallingAgent
      │  think → call tool → observe → … → final answer
      ▼
┌──────────────────────┬──────────────────────────────────────────────────────────────┐
│ list_values          │ discover the exact vocabulary (drug names, histologies, biomarker values) │
│ filter_models        │ structured filters: cancer system, histology, site, sex, tumour type,      │
│                      │ biomarkers, drug, response, treatment source, paediatric, min. drug tests  │
│ semantic_search      │ hybrid BM25 + dense (MiniLM) search with RRF, reused from OncoRAG          │
│ get_model_details    │ the full record of one model                                               │
└──────────────────────┴──────────────────────────────────────────────────────────────┘
      │
      ▼
Gradio chat UI that shows every tool call, its arguments and its result
```

The model, endpoint and key live in `.env` (`LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`), so any OpenAI-compatible provider can be swapped in.

## Evaluation

`evaluate_agent.py` runs 15 requests whose correct answer is known. The **ground truth is computed by applying the same filters directly to the data, with no LLM involved**. Then it checks:

- **ID precision:** of the models the agent recommends, how many actually meet every criterion?
- **"No result" handling:** when nothing matches (an invented drug, an impossible cancer, a fake ID), does the agent say so instead of inventing models?
- **Hallucinated links:** any accession that does not exist in the catalogue.
- **Cost:** steps and latency per request.

The test set covers easy, multi-criteria and paediatric requests, a sparse field (breast biomarkers, about 3% filled), patient-received versus PDX-tested drugs, and 4 traps.

### Results across iterations

| Version | Change | Result |
|---|---|---|
| v1 | First run | A tool crashed on empty drug/response results, and the agent burned all its steps retrying |
| v2 | Crash fixed | 12/15. One real agent error (HER2) and two evaluation-metric false negatives |
| **v3** | Tools return the **real field values** as evidence, better tool docstrings, metric fixed | **15/15** (14 scored automatically, plus 1 metric false negative verified by reading the answer) · **ID precision 1.00** · **0 hallucinated links** · 3.3 steps on average |

**The LLM never changed.** Every improvement came from the tools, the instructions and the evaluation.

### Failure modes found (and fixed)

1. **Claiming an unverified property.** The agent reported "126 pancreatic models *with drug tests*" without filtering on drug tests (the true number is 16). Fixed with a `min_drug_tests` filter and the instruction *"never claim a property you did not filter or verify."*
2. **Concluding absence when the tool could not check.** The agent answered "there are no paediatric nervous-system models" because the tool had no paediatric filter. There are 149. Fixed by exposing the `paediatric` field. *Absence of evidence is not evidence of absence.*
3. **Ambiguous field name led to a wrong claim.** `n_model_drug_tests` was read as "tests with paclitaxel", when it counts tests with any drug. Renamed to `n_drug_tests_in_model_ALL_DRUGS`, and the tool now adds an explicit note.
4. **Substring filter plus invented evidence.** Filtering `breast_biomarkers="HER2"` also matched *HER2 negative* (111 models instead of 3), and the agent quoted a field value that does not exist. Fixed by returning the **real value** of every filtered field, so evidence comes from the data and not from the model, and by documenting the value format in the tool description.
5. **Over-filtering.** The agent sometimes adds criteria nobody asked for, such as `cancer_system="Digestive System"` for pancreas. That did not change the result here, but it could silently exclude models labelled "Unclassified" (663 models).

## Limitations

- **Metadata only.** Gene-level mutation calls and expression values live in per-model data files and are not integrated yet, so requests such as "PIK3CA-mutant models" are out of scope.
- **Drug responses are dominated by "Progressive Disease"** (about 83% of PDX tests, including untreated and saline controls). Complete responses are rare (78), and the agent is instructed not to overstate them.
- **Small evaluation set** (15 requests, single run). LLM outputs are not deterministic, so results can vary between runs.
- **Latency.** Most steps take 0.3–1.5 s, but some API calls queue for about 55 s on the provider's free tier.
- **Free-text answers are scored with simple heuristics** (model ID matching and negative-phrase detection), so reading `results/eval_answers.md` remains necessary.

## Repository

```
fetch_metadata.py    # downloads 10,348 model records from BioStudies (parallel, cached, resumable)
build_dataset.py     # PDX only: cleaning, lists of treatments, Pydantic validation, text descriptions
retrievers.py        # BM25 / dense / hybrid (RRF) search, reused from OncoRAG
tools.py             # the 4 agent tools (testable without an LLM: python tools.py)
agent.py             # smolagents ToolCallingAgent + instructions (CLI: python agent.py "question")
app.py               # Gradio chat UI showing every agent step
evaluate_agent.py    # 15-case evaluation with automatic ground truth
results/             # eval_results.csv, eval_answers.md
```

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate                 # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# .env  ->  LLM_BASE_URL=https://api.cerebras.ai/v1   LLM_MODEL=gpt-oss-120b   LLM_API_KEY=...
python fetch_metadata.py               # ~5 min
python build_dataset.py
python tools.py                        # first run builds the search index (~1 min)
python app.py                          # http://127.0.0.1:7860
python evaluate_agent.py               # ~5–10 min
```

## Stack

smolagents (ToolCallingAgent) · gpt-oss-120b via an OpenAI-compatible API · Gradio · pandas · Pydantic · sentence-transformers · bm25s · EMBL-EBI BioStudies API