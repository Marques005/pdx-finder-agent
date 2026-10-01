"""
explore_api.py · Passo 1 do P3: conhecer os dados do CancerModels.org antes de os descarregar
"""
import requests
import pandas as pd

BASE = "https://cancermodels.org/api"
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 40)

ENDPOINTS = {
    "model_metadata":             {"type": "eq.xenograft"},      # dados clínicos de cada modelo PDX
    "mutation_data_extended":     {"hgnc_symbol": "eq.PIK3CA"},  # mutações (aqui: só o gene PIK3CA)
    "drug_dosing_extended":       {},                            # resposta do modelo PDX a fármacos
    "patient_treatment_extended": {},                            # tratamentos que o DOENTE recebeu
}

for name, filters in ENDPOINTS.items():
    print("=" * 100)
    print(f"TABELA: {name}")
    try:
        r = requests.get(f"{BASE}/{name}", params={**filters, "limit": 3}, timeout=60)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        print(f"Colunas ({len(df.columns)}): {list(df.columns)}\n")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"ERRO: {e}")

print("=" * 100)
r = requests.get(f"{BASE}/model_metadata", params={"type": "eq.xenograft", "select": "model_id", "limit": 1},
                 headers={"Prefer": "count=exact"}, timeout=60)
print("Total de modelos PDX (xenograft):", r.headers.get("Content-Range", "?").split("/")[-1])