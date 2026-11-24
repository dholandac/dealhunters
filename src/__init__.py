"""Pacote principal da arquitetura modular do chatbot Deal Hunters."""

from src.api_client import CheapSharkClient, ExchangeRateClient
from src.business_logic import BusinessLogic
from src.chatbot import DealHuntersBot
from src.database import Database
from src.nlg import NLG
from src.nlu import NLU

__all__ = [
    "CheapSharkClient",
    "ExchangeRateClient",
    "BusinessLogic",
    "DealHuntersBot",
    "Database",
    "NLG",
    "NLU",
]
