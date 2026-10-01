"""
explore_biostudies.py · Ver como o BioStudies guarda um modelo do CancerModels.org
"""
import json
import requests

API = "https://www.ebi.ac.uk/biostudies/api/v1"

# 1. Pesquisa na coleção CancerModels.Org (primeiros 3 resultados com "PDX")
r = requests.get(f"{API}/CancerModels.Org/search", params={"query": "PDX", "pageSize": 3}, timeout=60)
print("PESQUISA:", r.status_code)
data = r.json()
print("Total de resultados:", data.get("totalHits"))
for hit in data.get("hits", [])[:3]:
    print(" -", hit.get("accession"), "|", hit.get("title"))

# 2. Um modelo completo: que informação tem?
acc = "S-CMO10361"
r = requests.get(f"{API}/studies/{acc}", timeout=60)
print(f"\nMODELO {acc}:", r.status_code)
study = r.json()
print("Chaves do topo:", list(study.keys()))
section = study.get("section", {})
print("\nAtributos do modelo:")
for a in section.get("attributes", []):
    print(f"  {a.get('name')}: {str(a.get('value'))[:80]}")
print("\nSubsecções:", [s.get("type") for s in section.get("subsections", []) if isinstance(s, dict)])

# 3. Ficheiros disponíveis
r = requests.get(f"{API}/studies/{acc}/info", timeout=60)
print("\nINFO:", r.status_code, json.dumps(r.json(), indent=1)[:800])