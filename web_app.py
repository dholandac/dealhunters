"""
Interface Web opcional do Deal Hunters (FastAPI) para hospedagem na internet.

Reutiliza integralmente a mesma arquitetura modular desacoplada (`DealHuntersBot`)
executada no terminal (`main.py`), demonstrando desacoplamento real entre núcleo
e canal de atendimento.

Execução local:
    uvicorn web_app:app --reload
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import config
from src.chatbot import DealHuntersBot

app = FastAPI(
    title=f"{config.APP_NAME} — Chatbot Corporativo API",
    description="Assistente virtual de ofertas de jogos digitais e suporte pós-venda.",
    version="1.0.0",
)

bot = DealHuntersBot(seed_on_start=True)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=600)
    session_token: Optional[str] = None


class ChatResponse(BaseModel):
    session_token: str
    reply: str
    estado_anterior: str
    estado_atual: str
    intencao: str
    confianca: float
    sentimento: str
    score_insatisfacao: float
    falhas_consecutivas: int
    transbordo_ativo: bool
    origem_dados: str
    debug: Dict[str, Any]


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "app": config.APP_NAME,
        "nlp_backend": bot.nlu.backend_nlp,
        "offline_mode": config.OFFLINE_MODE,
    }


@app.post("/api/session")
def criar_sessao() -> Dict[str, str]:
    token, boas_vindas = bot.iniciar_sessao(session_token=str(uuid.uuid4()))
    return {"session_token": token, "welcome": boas_vindas}


@app.post("/api/chat", response_model=ChatResponse)
def enviar_mensagem(payload: ChatRequest) -> ChatResponse:
    token = payload.session_token or str(uuid.uuid4())
    resp = bot.responder(payload.message, session_token=token)
    return ChatResponse(
        session_token=resp.session_token,
        reply=resp.resposta,
        estado_anterior=resp.estado_anterior,
        estado_atual=resp.estado_atual,
        intencao=resp.intencao,
        confianca=resp.confianca,
        sentimento=resp.sentimento,
        score_insatisfacao=resp.score_insatisfacao,
        falhas_consecutivas=resp.falhas_consecutivas,
        transbordo_ativo=resp.transbordo_ativo,
        origem_dados=resp.origem_dados,
        debug=resp.debug_info,
    )


@app.get("/api/session/{session_token}/logs")
def obter_logs_lgpd(session_token: str) -> Dict[str, Any]:
    logs = bot.db.listar_logs_sessao(session_token=session_token, limite=30)
    return {
        "session_token": session_token,
        "total": len(logs),
        "logs": [
            {
                "id": log.id,
                "remetente": log.remetente,
                "mensagem_sanitizada": log.mensagem_sanitizada,
                "intencao": log.intencao_detectada,
                "confianca": log.confianca_nlu,
                "sentimento": log.sentimento,
                "estado": log.estado_apos_turno,
                "anonimizado_lgpd": log.anonimizado_lgpd,
                "timestamp": log.timestamp.isoformat(),
            }
            for log in logs
        ],
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Deal Hunters — Assistente Virtual Hunter</title>
  <style>
    :root {
      --bg: #0f172a;
      --panel: #1e293b;
      --border: #334155;
      --accent: #38bdf8;
      --accent-2: #22c55e;
      --text: #f8fafc;
      --muted: #94a3b8;
      --user-bg: #0284c7;
      --bot-bg: #1e293b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at top, #172554 0%, var(--bg) 55%);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      padding: 20px;
    }
    .layout {
      width: 100%;
      max-width: 1080px;
      display: grid;
      grid-template-columns: 1fr 320px;
      gap: 16px;
      height: calc(100vh - 40px);
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; height: auto; }
      .debug-card { min-height: 260px; }
    }
    .card {
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid var(--border);
      border-radius: 16px;
      display: flex;
      flex-direction: column;
      overflow: hidden;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.4);
    }
    .header {
      padding: 16px 20px;
      background: rgba(30, 41, 59, 0.9);
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .title h1 { font-size: 1.15rem; color: var(--accent); }
    .title p { font-size: 0.82rem; color: var(--muted); margin-top: 2px; }
    .btn-small {
      background: transparent;
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.8rem;
    }
    .btn-small:hover { border-color: var(--accent); color: var(--accent); }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      padding: 10px 16px;
      border-bottom: 1px solid var(--border);
      background: rgba(15, 23, 42, 0.5);
    }
    .chip {
      background: #1e293b;
      color: #cbd5e1;
      border: 1px solid #334155;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 0.78rem;
      cursor: pointer;
    }
    .chip:hover { border-color: var(--accent); color: #fff; }
    .messages {
      flex: 1;
      overflow-y: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .msg {
      max-width: 85%;
      padding: 12px 14px;
      border-radius: 14px;
      white-space: pre-wrap;
      line-height: 1.48;
      font-size: 0.92rem;
    }
    .msg.user {
      align-self: flex-end;
      background: var(--user-bg);
      color: #fff;
      border-bottom-right-radius: 4px;
    }
    .msg.bot {
      align-self: flex-start;
      background: var(--bot-bg);
      border: 1px solid var(--border);
      border-bottom-left-radius: 4px;
    }
    .input-row {
      display: flex;
      gap: 10px;
      padding: 14px 16px;
      border-top: 1px solid var(--border);
      background: rgba(30, 41, 59, 0.75);
    }
    .input-row input {
      flex: 1;
      background: #0f172a;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 11px 14px;
      color: var(--text);
      font-size: 0.94rem;
      outline: none;
    }
    .input-row input:focus { border-color: var(--accent); }
    .input-row button {
      background: var(--accent);
      color: #082f49;
      font-weight: 700;
      border: none;
      border-radius: 10px;
      padding: 0 18px;
      cursor: pointer;
    }
    .input-row button:disabled { opacity: 0.5; cursor: not-allowed; }
    .debug-body {
      padding: 14px;
      overflow-y: auto;
      font-size: 0.82rem;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .kv {
      background: #1e293b;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
    }
    .kv label {
      display: block;
      font-size: 0.72rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .kv code {
      color: #e2e8f0;
      word-break: break-word;
      font-family: Consolas, monospace;
    }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
      background: rgba(56, 189, 248, 0.16);
      color: var(--accent);
    }
  </style>
</head>
<body>
  <div class="layout">
    <section class="card">
      <div class="header">
        <div class="title">
          <h1>Deal Hunters — Assistente Hunter</h1>
          <p>Preços CheapShark em tempo real + Pós-venda Stateful (SQLAlchemy + LGPD)</p>
        </div>
        <button class="btn-small" onclick="resetSession()">Nova Sessão</button>
      </div>
      <div class="chips">
        <button class="chip" onclick="quickSend('Quanto custa Hollow Knight?')">Preço Hollow Knight</button>
        <button class="chip" onclick="quickSend('tem promo massa ate 40 conto na steam?')">Gírias + Orçamento</button>
        <button class="chip" onclick="quickSend('Quero consultar meu pedido DH-1001')">Pedido DH-1001</button>
        <button class="chip" onclick="quickSend('Me avisa quando baixar para 30 reais')">Criar Alerta</button>
        <button class="chip" onclick="quickSend('Paguei e a chave deu invalida, absurdo, quero reembolso urgente!')">Urgência / Transbordo</button>
      </div>
      <div id="messages" class="messages"></div>
      <form class="input-row" onsubmit="sendMessage(event)">
        <input id="userInput" type="text" placeholder="Digite sua mensagem para o Hunter..." autocomplete="off" />
        <button id="sendBtn" type="submit">Enviar</button>
      </form>
    </section>

    <aside class="card debug-card">
      <div class="header">
        <div class="title">
          <h1>Telemetria NLU &amp; Estado</h1>
          <p>Inspeção ao vivo da arquitetura modular</p>
        </div>
      </div>
      <div class="debug-body" id="debugPanel">
        <div class="kv"><label>Status</label><code>Iniciando sessão...</code></div>
      </div>
    </aside>
  </div>

  <script>
    let sessionToken = null;
    const messagesEl = document.getElementById('messages');
    const debugEl = document.getElementById('debugPanel');
    const inputEl = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    function addMsg(text, sender) {
      const div = document.createElement('div');
      div.className = 'msg ' + sender;
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    function renderDebug(data) {
      debugEl.innerHTML = `
        <div class="kv"><label>SessaoUsuario (Estado)</label><span class="badge">${data.estado_anterior} → ${data.estado_atual}</span></div>
        <div class="kv"><label>Intenção &amp; Confiança</label><code>${data.intencao} (${(data.confianca * 100).toFixed(0)}%)</code></div>
        <div class="kv"><label>Entidades (NER)</label><code>${JSON.stringify(data.debug.entidades || {})}</code></div>
        <div class="kv"><label>Lemas (spaCy/NLTK)</label><code>${JSON.stringify(data.debug.lemas || [])}</code></div>
        <div class="kv"><label>Sentimento / Fallbacks</label><code>${data.sentimento} (${data.score_insatisfacao}) | Fallbacks: ${data.falhas_consecutivas}/2</code></div>
        <div class="kv"><label>Origem dos Dados</label><code>${data.origem_dados} | ${data.debug.backend_nlp || ''}</code></div>
      `;
    }

    async function resetSession() {
      messagesEl.innerHTML = '';
      const res = await fetch('/api/session', { method: 'POST' });
      const data = await res.json();
      sessionToken = data.session_token;
      addMsg(data.welcome, 'bot');
      debugEl.innerHTML = `<div class="kv"><label>Sessão Ativa</label><code>${sessionToken}</code></div>`;
      inputEl.focus();
    }

    function quickSend(text) {
      inputEl.value = text;
      sendMessage(new Event('submit'));
    }

    async function sendMessage(e) {
      e.preventDefault();
      const text = inputEl.value.trim();
      if (!text) return;
      addMsg(text, 'user');
      inputEl.value = '';
      sendBtn.disabled = true;
      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, session_token: sessionToken })
        });
        const data = await res.json();
        sessionToken = data.session_token;
        addMsg(data.reply, 'bot');
        renderDebug(data);
      } catch (err) {
        addMsg('Erro de comunicação com o servidor Deal Hunters.', 'bot');
      } finally {
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    resetSession();
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    return HTMLResponse(content=HTML_PAGE)
