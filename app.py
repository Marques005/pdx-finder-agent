"""
app.py · Passo 6 do P3: interface conversacional (Gradio)

Usa a interface de chat do próprio smolagents, que mostra em tempo real cada passo do agente:
que ferramenta chamou, com que argumentos e o que recebeu. É a "apresentação transparente" que a vaga pede.

Correr localmente:  python app.py   -> abre http://127.0.0.1:7860 no browser
"""
from smolagents import GradioUI
from agent import build_agent

agent, _ = build_agent(verbosity=0)

try:
    ui = GradioUI(agent, reset_agent_memory=True)   # cada pergunta começa do zero
except TypeError:
    ui = GradioUI(agent)

if __name__ == "__main__":
    ui.launch()