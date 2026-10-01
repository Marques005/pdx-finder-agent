"""
evaluate_agent.py · Passo 7 do P3: avaliar o agente com perguntas cuja resposta certa conhecemos

Para cada pergunta definimos a "verdade" com os MESMOS filtros aplicados diretamente aos dados
(sem LLM). Depois corremos o agente e medimos:

  - precisão dos IDs   : dos modelos que o agente recomendou, quantos cumprem MESMO os critérios?
  - acerto no "nenhum" : quando não existe nenhum modelo, o agente diz isso (e não inventa)?
  - alucinações        : links/acessos citados que não existem na base de dados
  - passos, tempo      : custo de cada resposta

As respostas completas ficam em results/eval_answers.md para leitura humana (avaliação qualitativa).
Usa:  python evaluate_agent.py
"""
import os, re, json, time
import pandas as pd
from tools import DF, filter_models
from agent import build_agent

os.makedirs("results", exist_ok=True)
ALL_IDS = set(DF.model_id)
ALL_ACC = set(DF.accession)

CASES = [
    # (id, pergunta, filtros que definem a resposta certa)
    ("breast_ptx_cr", "Quero modelos de cancro da mama com resposta completa ao paclitaxel.",
     dict(cancer_system="Breast", drug="Paclitaxel", response="Complete Response")),
    ("panc_female_tested", "Preciso de modelos de cancro do pâncreas de doentes do sexo feminino com testes de fármacos.",
     dict(primary_site="pancreas", patient_sex="female", min_drug_tests=1)),
    ("melanoma_bini_resp", "Que modelos de melanoma foram testados com binimetinib e tiveram resposta parcial ou completa?",
     dict(histology="Melanoma", drug="Binimetinib", response="Response")),
    ("crc_cetux_resp", "Modelos com histologia 'Colorectal Carcinoma' que tiveram resposta parcial ou completa ao cetuximab.",
     dict(histology="Colorectal Carcinoma", drug="Cetuximab", response="Response")),
    ("breast_her2", "Há modelos de cancro da mama HER2 positivo?",
     dict(breast_biomarkers="HER2/ERBB2 positive")),
    ("paed_nervous", "Preciso de modelos pediátricos de tumores do sistema nervoso.",
     dict(cancer_system="Nervous System", paediatric=True)),
    ("paed_neuroblastoma_tested", "Há modelos pediátricos de neuroblastoma com testes de fármacos no modelo?",
     dict(histology="Neuroblastoma", paediatric=True, min_drug_tests=1)),
    ("lung_scc_male", "Modelos de carcinoma pulmonar de células escamosas de doentes do sexo masculino.",
     dict(histology="Lung Squamous Cell Carcinoma", patient_sex="male")),
    ("panc_metastatic", "Quero modelos de adenocarcinoma do pâncreas de tumores metastáticos.",
     dict(histology="Pancreatic", tumour_type="Metastatic")),
    ("aml_recurrent", "Modelos de leucemia mieloide aguda de tumores recorrentes.",
     dict(histology="Acute Myeloid Leukemia", tumour_type="Recurrent")),
    ("patient_cisplatin", "Que modelos vêm de doentes que receberam cisplatina como tratamento?",
     dict(drug="Cisplatin", treatment_source="patient")),
    ("fake_drug", "Há modelos tratados com Inventadomab?",
     dict(drug="Inventadomab")),
    ("fake_cancer", "Quero modelos de cancro do apêndice vermiforme em doentes com mais de 150 anos.",
     dict(primary_site="appendix", histology="ZZZ_impossible")),
    ("details_known", "Mostra-me os detalhes do modelo NIBRX-4567.", "NIBRX-4567"),
    ("details_fake", "Mostra-me os detalhes do modelo XYZ-0000.", None),
]


def ground_truth(spec):
    if spec is None:
        return set()
    if isinstance(spec, str):
        return {spec}
    res = json.loads(filter_models(**spec, limit=100000))
    return {m["model_id"] for m in res["models"]}


def mentioned_ids(text):
    found = set()
    for mid in ALL_IDS:
        if mid in text and re.search(rf"(?<![\w-]){re.escape(mid)}(?![\w-])", text):
            found.add(mid)
    return found


NEG = re.compile(r"\b(nenhum|nenhuma|não (há|existe|existem|foram encontrad|encontr)|no models|not found|0 modelos)", re.I)

rows, md = [], ["# Respostas do agente na avaliação\n"]
for cid, question, spec in CASES:
    truth = ground_truth(spec)
    agent, prepend = build_agent(verbosity=0)
    t0 = time.time()
    try:
        answer = str(agent.run(question))
        err = ""
    except Exception as e:
        answer, err = "", str(e)[:200]
    secs = time.time() - t0
    steps = sum(1 for s in getattr(agent.memory, "steps", []) if type(s).__name__ == "ActionStep")

    ids = mentioned_ids(answer)
    accs = set(re.findall(r"S-CMO\d+", answer))
    hallucinated = sorted(a for a in accs if a not in ALL_ACC)
    wrong_ids = sorted(ids - truth)
    if truth:
        precision = (len(ids & truth) / len(ids)) if ids else 0.0
        verdict = "ok" if ids and precision == 1.0 else ("parcial" if ids and precision > 0 else "falhou")
    else:
        precision = None
        verdict = "ok" if (not ids and NEG.search(answer)) else "falhou"

    rows.append(dict(case=cid, n_truth=len(truth), n_mentioned=len(ids), precision=precision,
                     wrong_ids=", ".join(wrong_ids[:5]), hallucinated=", ".join(hallucinated),
                     verdict=verdict, steps=steps, seconds=round(secs, 1), error=err))
    md += [f"## {cid}  ·  {verdict}", f"**Pergunta:** {question}",
           f"**Verdade:** {len(truth)} modelos  ·  **Citados:** {len(ids)}  ·  **Errados:** {wrong_ids[:5]}  ·  "
           f"**Alucinados:** {hallucinated}  ·  {steps} passos, {secs:.1f}s", "", answer, "\n---\n"]
    print(f"{cid:28s} {verdict:8s} truth={len(truth):4d} cited={len(ids):2d} "
          f"prec={'—' if precision is None else f'{precision:.2f}'} steps={steps} {secs:5.1f}s {err}")
    time.sleep(2)  # pausa entre perguntas (limites da API)

df = pd.DataFrame(rows)
df.to_csv("results/eval_results.csv", index=False)
open("results/eval_answers.md", "w", encoding="utf-8").write("\n".join(md))

print("\n================ RESUMO ================")
print(df["verdict"].value_counts().to_string())
with_truth = df[df.n_truth > 0]
print(f"\nPrecisão média dos IDs (perguntas com resposta): {with_truth.precision.mean():.2f}")
print(f"Perguntas sem resposta tratadas corretamente:   {(df[df.n_truth == 0].verdict == 'ok').sum()} / {(df.n_truth == 0).sum()}")
print(f"Acessos alucinados no total:                     {sum(bool(h) for h in df.hallucinated)}")
print(f"Passos médios: {df.steps.mean():.1f}  ·  Tempo médio: {df.seconds.mean():.1f}s")
print("\nRespostas completas em results/eval_answers.md")