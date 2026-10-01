"""
explore_biostudies2.py · Descobrir (1) como listar todos os modelos e (2) que dados tem cada um
"""
import json
import requests

API = "https://www.ebi.ac.uk/biostudies/api/v1"


def show_search(url, params):
    try:
        r = requests.get(url, params=params, timeout=60)
        d = r.json()
        hits = d.get("hits") or []
        print(f"{r.status_code} | total={d.get('totalHits')} | {url.split('/v1/')[1]} {params}")
        if hits:
            print("   1.º resultado:", {k: hits[0].get(k) for k in ("accession", "title", "type")})
        return d
    except Exception as e:
        print("ERRO", url, params, e)
        return {}


print("========== 1. Como pesquisar a coleção ==========")
show_search(f"{API}/CancerModels.Org/search", {"pageSize": 3})
show_search(f"{API}/cancermodels.org/search", {"pageSize": 3})
show_search(f"{API}/search", {"query": "CancerModels.Org", "pageSize": 3})
show_search(f"{API}/search", {"query": "PDX Acute Myeloid Leukemia", "pageSize": 3})
d = show_search(f"{API}/CancerModels.Org/search", {"query": "leukemia", "pageSize": 3})
print("   chaves da resposta:", list(d.keys()))

print("\n========== 2. Dados do doente/tumor do modelo S-CMO10361 ==========")
study = requests.get(f"{API}/studies/S-CMO10361", timeout=60).json()


def walk(section, depth=0):
    """Percorre as subsecções e mostra os atributos de cada uma."""
    for sub in section.get("subsections", []):
        items = sub if isinstance(sub, list) else [sub]
        for s in items:
            print("  " * depth + f"[{s.get('type')}]")
            for a in s.get("attributes", []):
                print("  " * depth + f"   {a.get('name')}: {str(a.get('value'))[:70]}")
            walk(s, depth + 1)


walk(study["section"])

print("\n========== 3. Procurar um modelo PDX com ficheiros de dados ==========")
for acc in ["S-CMO1", "S-CMO100", "S-CMO1000", "S-CMO2000", "S-CMO3000", "S-CMO5000"]:
    try:
        info = requests.get(f"{API}/studies/{acc}/info", timeout=60).json()
        st = requests.get(f"{API}/studies/{acc}", timeout=60).json()
        attrs = {a["name"]: a["value"] for a in st["section"].get("attributes", [])}
        print(f"{acc}: files={info.get('files')} | {attrs.get('Model type')} | {attrs.get('Histology')} | dataset={attrs.get('Dataset Available')}")
        if info.get("files"):
            print("   pasta:", info.get("httpLink"))
    except Exception as e:
        print(acc, "ERRO", e)