"""
fetch_metadata.py · Passo 2 do P3: descarregar os metadados de TODOS os modelos do CancerModels.org (via BioStudies)

- Os modelos têm acessos sequenciais: S-CMO1, S-CMO2, ..., ~S-CMO10400
- Cada resposta é guardada em data/raw/<acesso>.json  -> se o script parar, continua de onde ficou (cache)
- No fim, junta tudo numa tabela: data/models_all.parquet (uma linha por modelo)
- Usa 6 pedidos em paralelo, para ser rápido sem sobrecarregar o servidor do EBI
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

API = "https://www.ebi.ac.uk/biostudies/api/v1"
MAX_ACC = 10400
WORKERS = 6
RAW = "data/raw"
os.makedirs(RAW, exist_ok=True)
session = requests.Session()


def fetch(acc, retries=4):
    path = f"{RAW}/{acc}.json"
    if os.path.exists(path):
        return "cache"
    for attempt in range(retries):
        try:
            st = session.get(f"{API}/studies/{acc}", timeout=60)
            if st.status_code == 404:
                json.dump({"missing": True}, open(path, "w"))
                return "404"
            st.raise_for_status()
            info = session.get(f"{API}/studies/{acc}/info", timeout=60).json()
            json.dump({"study": st.json(), "info": info}, open(path, "w", encoding="utf-8"))
            return "ok"
        except Exception:
            time.sleep(2 ** attempt)
    return "erro"


def flatten(acc, d):
    """Transforma a estrutura aninhada do BioStudies numa linha plana."""
    sec = d["study"]["section"]
    row = {"accession": acc}
    for a in sec.get("attributes", []):
        row[a["name"]] = a.get("value")

    def walk(s):
        for sub in s.get("subsections", []):
            for x in (sub if isinstance(sub, list) else [sub]):
                t = x.get("type", "")
                for a in x.get("attributes", []):
                    row[f"{t} | {a['name']}"] = a.get("value")
                walk(x)

    walk(sec)
    row["n_files"] = d["info"].get("files", 0)
    row["files_url"] = d["info"].get("httpLink")
    return row


if __name__ == "__main__":
    accs = [f"S-CMO{i}" for i in range(1, MAX_ACC + 1)]
    t0, counts = time.time(), {}
    with ThreadPoolExecutor(WORKERS) as ex:
        futures = {ex.submit(fetch, a): a for a in accs}
        for n, f in enumerate(as_completed(futures), start=1):
            s = f.result()
            counts[s] = counts.get(s, 0) + 1
            if n % 250 == 0:
                print(f"{n}/{len(accs)} | {counts} | {time.time() - t0:.0f}s")
    print("Descarga terminada:", counts)

    rows = []
    for a in accs:
        d = json.load(open(f"{RAW}/{a}.json", encoding="utf-8"))
        if "study" in d:
            rows.append(flatten(a, d))
    df = pd.DataFrame(rows)
    df.to_parquet("data/models_all.parquet", index=False)

    print(f"\n{len(df)} modelos, {df.shape[1]} colunas")
    print("\nTipos de modelo:\n", df["Model type"].value_counts().to_string())
    print("\nDataset Available (top 10):\n", df["Dataset Available"].value_counts().head(10).to_string())
    print("\nColunas:\n", list(df.columns))