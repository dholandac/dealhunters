"""
Configurações centralizadas do projeto Deal Hunters.
Permite sobrescrever parâmetros via variáveis de ambiente (.env).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Identidade corporativa do chatbot
APP_NAME = os.getenv("APP_NAME", "Deal Hunters")
BOT_NAME = os.getenv("BOT_NAME", "Hunter")
COMPANY_SLOGAN = "Sua central inteligente de ofertas e suporte em jogos digitais"

# Banco de dados (Eixo 3: SQLAlchemy + SQLite)
DEFAULT_DB_PATH = BASE_DIR / "dealhunters.db"
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# APIs REST externas (Eixo 2: requests)
CHEAPSHARK_BASE_URL = os.getenv(
    "CHEAPSHARK_BASE_URL", "https://www.cheapshark.com/api/1.0"
)
CHEAPSHARK_REDIRECT_TEMPLATE = (
    "https://www.cheapshark.com/redirect?dealID={deal_id}"
)
CHEAPSHARK_USER_AGENT = os.getenv(
    "CHEAPSHARK_USER_AGENT",
    "DealHuntersBot/1.0 (Projeto Academico Chatbot Python; contato@dealhunters.local)",
)

EXCHANGE_RATE_URL = os.getenv(
    "EXCHANGE_RATE_URL", "https://economia.awesomeapi.com.br/json/last/USD-BRL"
)
DEFAULT_USD_BRL_RATE = float(os.getenv("DEFAULT_USD_BRL_RATE", "5.80"))
EXCHANGE_CACHE_TTL_SECONDS = int(os.getenv("EXCHANGE_CACHE_TTL_SECONDS", "1800"))

API_TIMEOUT_SECONDS = float(os.getenv("API_TIMEOUT_SECONDS", "8.0"))
API_MAX_RETRIES = int(os.getenv("API_MAX_RETRIES", "2"))

# Modo offline opcional (útil para testes ou contingência de rede em laboratório)
OFFLINE_MODE = os.getenv("DEALHUNTERS_OFFLINE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "sim",
}
FALLBACK_CATALOG_PATH = DATA_DIR / "fallback_catalog.json"

# NLU / NLP (Eixo 1)
SPACY_MODEL_NAME = os.getenv("SPACY_MODEL_NAME", "pt_core_news_sm")
NLU_CONFIDENCE_THRESHOLD = float(os.getenv("NLU_CONFIDENCE_THRESHOLD", "0.35"))

# Regras de Negócio e Transbordo (Eixo 4)
MAX_CONSECUTIVE_FALLBACKS = int(os.getenv("MAX_CONSECUTIVE_FALLBACKS", "2"))
URGENCY_SCORE_THRESHOLD = float(os.getenv("URGENCY_SCORE_THRESHOLD", "0.65"))

# Servidor Web opcional (FastAPI)
WEB_HOST = os.getenv("HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("PORT", "8000"))
