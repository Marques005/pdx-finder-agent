"""
build_dataset.py · Passo 3 do P3: juntar os dados heterogéneos numa representação única

Lê os 10 348 JSON de data/raw/ (já descarregados) e cria, SÓ para os modelos PDX:
  - data/pdx_models.parquet : uma linha por modelo, com campos limpos
  - cada modelo tem LISTAS de tratamentos (um modelo pode ter vários) e um texto descritivo
    (o texto vai servir para a pesquisa por significado, reutilizando o P2)
Valida cada modelo com Pydantic: se faltar algo essencial, o modelo é rejeitado e contado.
"""
import json, os, glob
from collections import Counter
from typing import Optional
import pandas as pd
from pydantic import BaseModel, field_validator

RAW = "data/raw"
MISSING = {"", "not provided", "not specified", "not collected", "n/a", "na", "unknown", "none"}


def clean(v):
    """Normaliza valores em falta para None."""
    if v is None:
        return None
    v = str(v).strip()
    return None if v.lower() in MISSING else v


class Treatment(BaseModel):
    drug: str
    dose: Optional[str] = None
    response: Optional[str] = None


class PDXModel(BaseModel):
    accession: str
    model_id: str
    histology: str
    cancer_system: Optional[str] = None
    primary_site: Optional[str] = None
    collection_site: Optional[str] = None
    tumour_type: Optional[str] = None
    cancer_stage: Optional[str] = None
    cancer_grade: Optional[str] = None
    patient_sex: Optional[str] = None
    patient_age: Optional[str] = None
    patient_ethnicity: Optional[str] = None
    paediatric: Optional[str] = None
    breast_biomarkers: Optional[str] = None
    provider: Optional[str] = None
    supplier_type: Optional[str] = None
    available_for_distribution: Optional[str] = None
    datasets_available: list[str] = []
    engraftment_site: Optional[str] = None
    engraftment_type: Optional[str] = None
    host_strain: Optional[str] = None
    qc_technique: Optional[str] = None
    tmb: Optional[str] = None
    msi: Optional[str] = None
    model_treatments: list[Treatment] = []      # resposta do MODELO PDX a fármacos (experiência no ratinho)
    patient_treatments: list[Treatment] = []    # tratamento que o DOENTE recebeu
    molecular_data_types: list[str] = []
    n_publications: int = 0
    n_files: int = 0
    url: str

    @field_validator("histology")
    @classmethod
    def histology_required(cls, v):
        if not v:
            raise ValueError("modelo sem histologia")
        return v


def collect_subsections(section):
    """Devolve {tipo_de_subsecção: [ {atributo: valor}, ... ]} percorrendo a árvore toda.
    Ao contrário da versão do fetch_metadata, guarda TODAS as entradas repetidas (ex.: vários tratamentos)."""
    out = {}

    def walk(s):
        for sub in s.get("subsections", []):
            for x in (sub if isinstance(sub, list) else [sub]):
                t = x.get("type") or "Other"
                attrs = {a["name"]: a.get("value") for a in x.get("attributes", [])}
                if attrs:
                    out.setdefault(t, []).append(attrs)
                walk(x)

    walk(section)
    return out


def treatments(entries, drug_key):
    out = []
    for e in entries:
        drug = clean(e.get(drug_key))
        if drug:
            out.append(Treatment(drug=drug, dose=clean(e.get("Dose")), response=clean(e.get("Response"))))
    return out


def build(acc, d):
    sec = d["study"]["section"]
    top = {a["name"]: a.get("value") for a in sec.get("attributes", [])}
    if top.get("Model type") != "PDX":
        return None
    subs = collect_subsections(sec)
    first = lambda t: subs.get(t, [{}])[0]
    pt, eng, qc = first("Patient / Tumour Metadata"), first("PDX model engraftment"), first("Model quality control")
    other = first("Other")
    ds = clean(top.get("Dataset Available"))
    return PDXModel(
        accession=acc,
        model_id=top.get("Model ID") or acc,
        histology=clean(top.get("Histology")) or "",
        cancer_system=clean(pt.get("Cancer System")),
        primary_site=clean(pt.get("Primary Site")),
        collection_site=clean(pt.get("Collection Site")),
        tumour_type=clean(pt.get("Tumour Type")),
        cancer_stage=clean(pt.get("Cancer Stage")),
        cancer_grade=clean(pt.get("Cancer Grade")),
        patient_sex=clean(pt.get("Patient Sex")),
        patient_age=clean(pt.get("Patient Age")),
        patient_ethnicity=clean(pt.get("Patient Ethnicity")),
        paediatric=clean(top.get("Paediatric")),
        breast_biomarkers=clean(top.get("Breast Cancer Biomarkers")),
        provider=clean(first("Organization").get("Name")),
        supplier_type=clean(top.get("Supplier type")),
        available_for_distribution=clean(top.get("Available for distribution")),
        datasets_available=[x.strip() for x in ds.split(",")] if ds else [],
        engraftment_site=clean(eng.get("Engraftment Site")),
        engraftment_type=clean(eng.get("Type")),
        host_strain=clean(eng.get("Host Strain Name")),
        qc_technique=clean(qc.get("Technique")),
        tmb=clean(other.get("TMB")),
        msi=clean(other.get("MSI")),
        model_treatments=treatments(subs.get("Model treatment", []), "Drug"),
        patient_treatments=treatments(subs.get("Patient treatment", []), "Treatment"),
        molecular_data_types=sorted({clean(m.get("Data Type")) for m in subs.get("Molecular data", []) if clean(m.get("Data Type"))}),
        n_publications=len(subs.get("Publications", [])),
        n_files=d["info"].get("files", 0) or 0,
        url=f"https://www.ebi.ac.uk/biostudies/CancerModels.Org/studies/{acc}",
    )


def describe(m: PDXModel) -> str:
    """Texto descritivo em inglês (como os dados) para a pesquisa por significado."""
    parts = [f"PDX model {m.model_id}: {m.histology}."]
    if m.cancer_system: parts.append(f"Cancer system: {m.cancer_system}.")
    if m.primary_site: parts.append(f"Primary site: {m.primary_site}.")
    if m.tumour_type: parts.append(f"Tumour type: {m.tumour_type}.")
    if m.cancer_stage: parts.append(f"Stage: {m.cancer_stage}.")
    if m.cancer_grade: parts.append(f"Grade: {m.cancer_grade}.")
    if m.patient_sex or m.patient_age: parts.append(f"Patient: {m.patient_sex or ''} {m.patient_age or ''}.".replace("  ", " "))
    if m.breast_biomarkers: parts.append(f"Breast biomarkers: {m.breast_biomarkers}.")
    if m.model_treatments:
        parts.append("Model drug responses: " + "; ".join(f"{t.drug} ({t.response or 'response n/a'})" for t in m.model_treatments[:10]) + ".")
    if m.patient_treatments:
        parts.append("Patient treatments: " + "; ".join(f"{t.drug} ({t.response or 'response n/a'})" for t in m.patient_treatments[:10]) + ".")
    if m.molecular_data_types: parts.append("Molecular data: " + ", ".join(m.molecular_data_types) + ".")
    if m.provider: parts.append(f"Provider: {m.provider}.")
    return " ".join(parts)


if __name__ == "__main__":
    models, rejected, errors = [], Counter(), 0
    for path in sorted(glob.glob(f"{RAW}/S-CMO*.json")):
        d = json.load(open(path, encoding="utf-8"))
        if "study" not in d:
            continue
        acc = os.path.basename(path)[:-5]
        try:
            m = build(acc, d)
            if m:
                models.append(m)
        except Exception as e:
            rejected[str(e).splitlines()[-1][:60]] += 1

    rows = []
    for m in models:
        r = m.model_dump()
        r["description"] = describe(m)
        r["n_model_treatments"] = len(m.model_treatments)
        r["n_patient_treatments"] = len(m.patient_treatments)
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_parquet("data/pdx_models.parquet", index=False)

    # ---------- Verificações ----------
    print(f"Modelos PDX válidos: {len(df)}  |  rejeitados: {sum(rejected.values())} {dict(rejected)}")
    print("\nSistemas de cancro (top 10):\n" + df["cancer_system"].value_counts().head(10).to_string())
    print("\nHistologias (top 10):\n" + df["histology"].value_counts().head(10).to_string())
    print(f"\nModelos com resposta a fármacos no PDX: {(df.n_model_treatments > 0).sum()}")
    print(f"Modelos com tratamento do doente:       {(df.n_patient_treatments > 0).sum()}")
    drugs = Counter(t["drug"] for ts in df["model_treatments"] for t in ts)
    print("\nFármacos mais testados nos modelos (top 10):", drugs.most_common(10))
    resp = Counter(t["response"] for ts in df["model_treatments"] for t in ts)
    print("Tipos de resposta:", resp.most_common(8))
    print("\nPercentagem de campos preenchidos:")
    print((df[["cancer_system", "primary_site", "tumour_type", "cancer_stage", "patient_sex", "patient_age", "breast_biomarkers", "provider"]].notna().mean() * 100).round(0).to_string())
    print("\nExemplo de descrição:\n", df.loc[df.n_model_treatments.idxmax(), "description"][:700])