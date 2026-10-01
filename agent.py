"""
agent.py · Passo 5 do P3: dar um "cérebro" às ferramentas

O agente segue o padrão ReAct (Reason + Act), em ciclo:
   1. PENSA: o que preciso de saber a seguir?
   2. AGE: chama uma ferramenta (ex.: filter_models com certos argumentos)
   3. OBSERVA: lê a resposta da ferramenta
   ... repete até ter informação suficiente ...
   4. RESPONDE: recomendação final, justificada com os dados

Usa:  python agent.py "a tua pergunta"
"""
import os, sys
from dotenv import load_dotenv
from smolagents import ToolCallingAgent

try:
    from smolagents import OpenAIServerModel as OpenAIModel
except ImportError:
    from smolagents import OpenAIModel

from tools import list_values, filter_models, semantic_search, get_model_details

load_dotenv(override=True)

INSTRUCTIONS = """You are a biobank assistant that helps cancer researchers select patient-derived xenograft (PDX)
models from the CancerModels.org catalogue (5,997 PDX models).

Rules:
- ALWAYS use the tools to answer. NEVER invent model IDs, drugs or responses.
- If you are unsure how a value is written (drug names, histologies, cancer systems), call list_values first.
- Use filter_models for precise criteria (cancer type, sex, drug, response). Use semantic_search for vague,
  descriptive requests, then verify the candidates with filter_models or get_model_details.
- Drug responses come from two sources: "model" = drug tested on the PDX in mice; "patient" = drug the patient received.
  Say which one you used.
- Most drug tests in the data are "Progressive Disease"; Complete/Partial responses are rare. Do not overstate results.
- If nothing matches, say so clearly and suggest how to relax the criteria.
- Never claim a property you did not filter or verify (e.g. "with drug tests" requires min_drug_tests=1).

Final answer format (in the SAME language as the user's question; if Portuguese, use European Portuguese, pt-PT):
1. One sentence: how many models matched and which criteria you applied.
2. Up to 5 recommended models. For each: model_id, histology, the evidence for EACH criterion
   (quote the field values), and a link https://www.ebi.ac.uk/biostudies/CancerModels.Org/studies/<accession>.
3. Caveats: missing data, small numbers, or assumptions you made."""


def build_agent(verbosity=1):
    model = OpenAIModel(
        model_id=os.environ["LLM_MODEL"],
        api_base=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    tools = [list_values, filter_models, semantic_search, get_model_details]
    try:
        return ToolCallingAgent(tools=tools, model=model, max_steps=8, verbosity_level=verbosity,
                                instructions=INSTRUCTIONS), False
    except TypeError:   # versões antigas do smolagents não têm 'instructions'
        return ToolCallingAgent(tools=tools, model=model, max_steps=8, verbosity_level=verbosity), True


def ask(question, verbosity=1):
    agent, prepend = build_agent(verbosity)
    task = f"{INSTRUCTIONS}\n\nUser question: {question}" if prepend else question
    return agent.run(task)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or ("Quero modelos PDX de cancro da mama que tenham tido resposta completa "
                                   "ao paclitaxel. Recomenda os 3 melhores e explica porquê.")
    print("\n" + "=" * 80 + "\nPERGUNTA: " + q + "\n" + "=" * 80)
    answer = ask(q)
    print("\n" + "=" * 80 + "\nRESPOSTA FINAL:\n" + str(answer))