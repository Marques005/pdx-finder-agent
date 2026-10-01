"""
tools.py · Passo 4 do P3: as FERRAMENTAS que o agente pode usar

O LLM nunca vê a tabela inteira. Em vez disso, chama estas funções (ferramentas) e recebe
respostas curtas em JSON. Cada ferramenta tem uma docstring clara: é ela que o LLM lê para
decidir QUANDO e COMO a usar.

  1. list_values         -> que valores existem num campo (ex.: nomes exatos de fármacos)
  2. filter_models       -> filtro estruturado (sistema, histologia, sexo, fármaco, resposta, pediátrico, ...)
  3. semantic_search     -> pesquisa por significado nas descrições (híbrido BM25 + embeddings, do P2)
  4. get_model_details   -> ficha completa de um modelo

Testar sem LLM:  python tools.py
"""
import json
from collections import Counter
from typing import Optional
import pandas as pd
from smolagents import tool

DF = pd.read_parquet("data/pdx_models.parquet")
DF["model_treatments"] = DF["model_treatments"].apply(lambda x: [dict(t) for t in x] if x is not None else [])
DF["patient_treatments"] = DF["patient_treatments"].apply(lambda x: [dict(t) for t in x] if x is not None else [])

RESPONSE_RANK = {"Complete Response": 0, "Partial Response": 1, "Stable Disease": 2, "Progressive Disease": 3}
TEXT_FIELDS = ["cancer_system", "histology", "primary_site", "tumour_type", "cancer_stage", "cancer_grade",
               "patient_sex", "patient_age", "breast_biomarkers", "provider", "available_for_distribution", "paediatric"]
_retriever = None


def _contains(series, value):
    return series.fillna("").str.contains(value, case=False, regex=False)


def _summary(row, drug=None, shown_fields=()):
    """Resumo curto de um modelo (para não encher o contexto do LLM).
    shown_fields: campos usados no filtro, cujo VALOR REAL é incluído como evidência (o LLM não o pode inventar)."""
    out = {"model_id": row.model_id, "accession": row.accession, "histology": row.histology,
           "primary_site": row.primary_site, "patient": f"{row.patient_sex or '?'}, {row.patient_age or '?'}",
           "n_drug_tests_in_model_ALL_DRUGS": int(row.n_model_treatments), "provider": row.provider}
    for f in shown_fields:
        out[f] = getattr(row, f)
    if drug:
        out["matching_treatments"] = [t for t in row.model_treatments + row.patient_treatments
                                      if drug.lower() in t["drug"].lower()][:5]
    return out


def _get_retriever():
    """Pesquisa híbrida (BM25 + embeddings gerais com RRF): a opção 'barata e boa' do P2."""
    global _retriever
    if _retriever is None:
        from retrievers import BM25Retriever, DenseRetriever, HybridRetriever
        docs = DF[["accession", "description"]].rename(columns={"accession": "pubid", "description": "text"})
        _retriever = HybridRetriever(
            [BM25Retriever(docs), DenseRetriever(docs, "sentence-transformers/all-MiniLM-L6-v2", name="pdx_general")],
            name="hybrid")
    return _retriever


@tool
def list_values(field: str, contains: Optional[str] = None, top: Optional[int] = 25) -> str:
    """Lists the most common values of a field, to find the EXACT vocabulary before filtering.
    Use it when you are unsure how something is written in the database (e.g. drug names, histologies).

    Args:
        field: one of cancer_system, histology, primary_site, tumour_type, cancer_stage, cancer_grade,
            patient_sex, patient_age, breast_biomarkers, provider, paediatric, model_drug, patient_drug, response.
        contains: optional case-insensitive text that values must contain (e.g. "breast", "taxel").
        top: maximum number of values to return (default 25).
    """
    if field in ("model_drug", "patient_drug", "response"):
        col = "patient_treatments" if field == "patient_drug" else "model_treatments"
        key = "response" if field == "response" else "drug"
        counts = Counter(t[key] for ts in DF[col] for t in ts if t.get(key))
    elif field in TEXT_FIELDS:
        counts = Counter(DF[field].dropna())
    else:
        return json.dumps({"error": f"unknown field '{field}'"})
    items = [(v, n) for v, n in counts.most_common() if not contains or contains.lower() in str(v).lower()]
    return json.dumps({"field": field, "values": [{"value": v, "n_models": n} for v, n in items[: top or 25]]})


@tool
def filter_models(cancer_system: Optional[str] = None, histology: Optional[str] = None,
                  primary_site: Optional[str] = None, tumour_type: Optional[str] = None,
                  patient_sex: Optional[str] = None, breast_biomarkers: Optional[str] = None,
                  drug: Optional[str] = None, response: Optional[str] = None,
                  treatment_source: Optional[str] = "model", min_drug_tests: Optional[int] = None,
                  paediatric: Optional[bool] = None, limit: Optional[int] = 10) -> str:
    """Filters PDX models by structured criteria. All text filters are case-insensitive 'contains'.
    Returns the total number of matches and a short summary of the best ones, including the REAL
    value of every field used as a filter (quote these values as evidence).

    Args:
        cancer_system: e.g. "Breast Cancer", "Digestive System", "Thoracic".
        histology: e.g. "Adenocarcinoma", "Melanoma", "Triple-Negative".
        primary_site: e.g. "lung", "pancreas", "large intestine".
        tumour_type: e.g. "Primary", "Metastatic", "Recurrent".
        patient_sex: "male" or "female".
        breast_biomarkers: text in the breast biomarker field. Values combine marker AND status, e.g.
            "HER2/ERBB2 positive" vs "HER2/ERBB2 negative": call list_values(field="breast_biomarkers") first
            and filter with the full value, never with just "HER2".
        drug: drug name (or part of it) the model was tested with or the patient received, e.g. "Cetuximab".
        response: response to that drug, e.g. "Complete Response", "Partial Response", "Stable Disease", "Progressive Disease".
        treatment_source: "model" (drug tested on the PDX in mice, default), "patient" (drug the patient received) or "any".
        min_drug_tests: only keep models with at least this many drug tests on the PDX (use 1 for "models with drug testing data").
        paediatric: True to keep only paediatric models (field 'paediatric' = Yes), False to exclude them.
        limit: maximum number of models to return (default 10).
    """
    m = pd.Series(True, index=DF.index)
    for col, val in [("cancer_system", cancer_system), ("histology", histology), ("primary_site", primary_site),
                     ("tumour_type", tumour_type), ("breast_biomarkers", breast_biomarkers)]:
        if val:
            m &= _contains(DF[col], val)
    if patient_sex:
        m &= DF["patient_sex"].fillna("").str.lower().eq(patient_sex.lower())
    if min_drug_tests:
        m &= DF["n_model_treatments"] >= min_drug_tests
    if paediatric is not None:
        m &= DF["paediatric"].fillna("").str.lower().eq("yes" if paediatric else "no")

    best_resp = pd.Series(9, index=DF.index)
    if drug or response:
        cols = {"model": ["model_treatments"], "patient": ["patient_treatments"]}.get(treatment_source or "model",
                                                                                      ["model_treatments", "patient_treatments"])

        def match(row):
            ranks = [RESPONSE_RANK.get(t.get("response"), 5) for c in cols for t in row[c]
                     if (not drug or drug.lower() in t["drug"].lower())
                     and (not response or response.lower() in (t.get("response") or "").lower())]
            return min(ranks) if ranks else None

        r = DF[m].apply(match, axis=1) if m.any() else pd.Series(dtype=float)
        found = r.dropna()
        m &= DF.index.isin(found.index)
        if len(found):
            best_resp.loc[found.index] = found.astype(int)

    # campos usados como filtro: o seu valor real vai na resposta, como evidência
    shown = [f for f, v in [("cancer_system", cancer_system), ("histology", histology), ("tumour_type", tumour_type),
                            ("breast_biomarkers", breast_biomarkers), ("paediatric", paediatric)] if v is not None]
    hits = DF[m].assign(_r=best_resp[m]).sort_values(["_r", "n_model_treatments"], ascending=[True, False])
    return json.dumps({"total_matches": int(m.sum()),
                       "note": "n_drug_tests_in_model_ALL_DRUGS counts tests with ANY drug, not only the requested one",
                       "showing": int(min(limit or 10, len(hits))),
                       "models": [_summary(row, drug, shown) for row in hits.head(limit or 10).itertuples()]},
                      default=str)


@tool
def semantic_search(query: str, k: Optional[int] = 10) -> str:
    """Free-text search over model descriptions (meaning-based + keyword hybrid search).
    Use it for vague or descriptive requests that do not map cleanly onto filter_models fields.

    Args:
        query: natural-language description of the desired models, in English (e.g. "metastatic melanoma resistant to MEK inhibitors").
        k: number of models to return (default 10).
    """
    res = _get_retriever().search([query], k=k or 10)[0]
    by_acc = DF.set_index("accession")
    return json.dumps({"models": [{**_summary(by_acc.loc[[acc]].reset_index().iloc[0]),
                                   "rank": i + 1,
                                   "snippet": by_acc.loc[acc, "description"][:220]} for i, (acc, _) in enumerate(res)]},
                      default=str)


@tool
def get_model_details(model_id: str) -> str:
    """Returns the full record of one PDX model (clinical, engraftment, treatments, data available, link).

    Args:
        model_id: the model_id (e.g. "NIBRX-5254") or the accession (e.g. "S-CMO5000").
    """
    row = DF[(DF.model_id == model_id) | (DF.accession == model_id)]
    if row.empty:
        return json.dumps({"error": f"model '{model_id}' not found"})
    rec = row.iloc[0].drop(labels=["description"]).to_dict()
    rec = {k: (list(v) if hasattr(v, "tolist") and not isinstance(v, str) else v) for k, v in rec.items()}
    rec = {k: (None if isinstance(v, float) and v != v else v) for k, v in rec.items()}   # NaN -> null
    rec["model_treatments"] = rec["model_treatments"][:15]
    rec["patient_treatments"] = rec["patient_treatments"][:15]
    return json.dumps(rec, default=str)


if __name__ == "__main__":
    print("--- list_values(model_drug, contains='taxel') ---")
    print(list_values(field="model_drug", contains="taxel"))
    print("\n--- filter_models(breast_biomarkers='HER2') -> deve mostrar o valor real (positive/negative) ---")
    print(filter_models(breast_biomarkers="HER2", limit=3)[:900])
    print("\n--- filter_models(cancer_system='Nervous System', paediatric=True) ---")
    print(filter_models(cancer_system="Nervous System", paediatric=True, limit=3)[:900])
    print("\n--- filter_models(histology='Colorectal Carcinoma', drug='Cetuximab', response='Complete Response') -> 0, sem erro ---")
    print(filter_models(histology="Colorectal Carcinoma", drug="Cetuximab", response="Complete Response"))
    print("\n--- semantic_search('metastatic melanoma treated with MEK inhibitor') ---")
    print(semantic_search(query="metastatic melanoma treated with MEK inhibitor", k=3)[:900])
    print("\n--- get_model_details('NIBRX-5254') ---")
    print(get_model_details(model_id="NIBRX-5254")[:600])