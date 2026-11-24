# Deal Hunters — Chatbot Corporativo de Ofertas e Suporte em Jogos Digitais

Projeto prático desenvolvido em **Python** com **Arquitetura Modular Desacoplada**.

O **Hunter**, assistente virtual oficial da loja de jogos digitais **Deal Hunters**, combina:
- **Eixo 1 — NLU/NLP (`spaCy` + suporte `NLTK`):** normalização de gírias/ruídos ortográficos, tokenização, remoção seletiva de stopwords, lematização, classificação de intenções via TF-IDF + Cosseno, NER híbrido (`EntityRuler` + extração dinâmica) e detecção de sentimento/urgência.
- **Eixo 2 — API REST (`requests`):** integração gratuita em tempo real com a **CheapShark API v1.0** (`/stores`, `/games`, `/deals`) e **AwesomeAPI** (`USD→BRL`), com retries, tratamento de exceções e fallback local de contingência.
- **Eixo 3 — Persistência Relacional e Estado (`SQLAlchemy` + `SQLite`):** controle **Stateful** em `SessaoUsuario`, auditoria com conformidade **LGPD** em `ConversaLog` (mascaramento automático de CPF/e-mail e anonimização sob demanda), além de `PedidoDigital` (validação por hash SHA-256), `AlertaPreco` e `TransbordoHumano`.
- **Eixo 4 — Regras de Negócio, Fallback e Transbordo Humano:** máquina de estados multi-turno, transbordo automático após **2 falhas consecutivas** ou **extrema insatisfação/urgência**, com protocolo gravado em banco de dados.

---

## Estrutura do Repositório

```text
dealhunters/
├── main.py                      # Arquivo mestre CLI (execução no terminal — Live Demo)
├── web_app.py                   # Interface Web opcional (FastAPI) para hospedagem online
├── config.py                    # Configurações centralizadas (.env)
├── requirements.txt             # Dependências Python
├── render.yaml / Procfile       # Configurações prontas para deploy gratuito na nuvem
├── data/
│   ├── seed_data.py             # Inicializa e pré-popula o banco SQLite (dealhunters.db)
│   └── fallback_catalog.json    # Catálogo local de contingência caso a API externa caia
├── src/
│   ├── database.py              # Classe Database + ORM SQLAlchemy + LGPD
│   ├── api_client.py            # Classes CheapSharkClient e ExchangeRateClient (requests)
│   ├── nlu.py                   # Classe NLU (spaCy, EntityRuler, NLTK, TF-IDF, Sentimento)
│   ├── business_logic.py        # Classe BusinessLogic (Máquina de Estados e Transbordo)
│   ├── nlg.py                   # Classe NLG (Persona Hunter)
│   └── chatbot.py               # Fachada DealHuntersBot
└── tests/
    └── test_chatbot.py          # Testes automatizados dos 3 cenários da Demo + LGPD + Rede
```

---

## Instalação e Execução Local

> **Requisito recomendado:** Python **3.11** ou **3.12** (melhor compatibilidade com `spaCy`).

```powershell
# 1. Acesse a pasta do projeto

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe o modelo em português do spaCy (e recursos opcionais do NLTK)
python -m spacy download pt_core_news_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('rslp')"

# 5. Crie e pré-popule o banco SQLite (dealhunters.db)
python data/seed_data.py
```

### Executar no Terminal (Live Demo obrigatória)

```powershell
# Modo normal (API CheapShark em tempo real)
python main.py

# Modo demonstração técnica (mostra lemas, NER, intenção, estado e origem da API)
python main.py --debug

# Modo contingência offline (demonstra resiliência sem internet)
python main.py --offline
```

### Executar Testes Automatizados

```powershell
pytest -v
```

### Executar a Interface Web Opcional (Local)

```powershell
uvicorn web_app:app --reload
```
Acesse `http://localhost:8000` no navegador.

---

## Dados Pré-Populados para Teste (Pós-venda)

| Código do Pedido | CPF de Teste | Jogo | Plataforma | Status |
|---|---|---|---|---|
| `DH-1001` | `111.222.333-44` | The Witcher 3: Wild Hunt | Steam | `ENTREGUE` (libera chave) |
| `DH-1002` | `555.666.777-88` | Elden Ring | Steam | `CHAVE_PENDENTE` |
| `DH-1003` | `999.888.777-66` | Hollow Knight | GOG | `AGUARDANDO_PAGAMENTO` |
| `DH-1004` | `111.222.333-44` | Cyberpunk 2077 | GOG | `REEMBOLSO_EM_ANALISE` |

---


