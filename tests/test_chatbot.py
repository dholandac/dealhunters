"""
Testes automatizados cobrindo os 4 Eixos Técnicos e os 3 Cenários da Live Demo:
  1. Fluxo padrão perfeito (multi-turno Stateful + CheapShark + Alerta + Pedido/LGPD)
  2. Mensagens com erros gramaticais, abreviações e gírias (robustez NLU/lematização)
  3. Acionamento de Fallback após 2 falhas consecutivas e Transbordo por urgência/insatisfação
  4. Resiliência contra exceções de rede da biblioteca requests (Timeout / ConnectionError)
  5. Conformidade LGPD (mascaramento de CPF/e-mail e anonimização de ConversaLog)

Execução:
    pytest -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.api_client import CheapSharkClient, IsThereAnyDealClient
from src.chatbot import DealHuntersBot
from src.database import Database


@pytest.fixture
def bot_isolado(tmp_path: Path) -> DealHuntersBot:
    """Instancia o chatbot com banco SQLite temporário e API em modo contingência determinístico."""
    db_file = tmp_path / "test_dealhunters.db"
    db = Database(database_url=f"sqlite:///{db_file.as_posix()}")
    api = CheapSharkClient(offline_mode=True)
    return DealHuntersBot(database=db, api_client=api, seed_on_start=True)


def test_cenario_1_fluxo_stateful_busca_preco_detalhe_e_alerta(
    bot_isolado: DealHuntersBot,
) -> None:
    token, _ = bot_isolado.iniciar_sessao()

    # Turno 1: Pede preço sem informar o jogo -> entra no estado AGUARDANDO_JOGO
    r1 = bot_isolado.responder("Quero ver o preço de um jogo", token)
    assert r1.intencao == "consultar_preco"
    assert r1.estado_atual == "AGUARDANDO_JOGO"

    # Turno 2: Informa o nome do jogo -> consulta ofertas e muda para EXIBINDO_OFERTAS
    r2 = bot_isolado.responder("Hollow Knight", token)
    assert r2.estado_atual == "EXIBINDO_OFERTAS"
    assert "Hollow Knight" in r2.resposta
    assert "Steam" in r2.resposta

    # Turno 3: Seleciona a opção 1 usando o contexto salvo na SessaoUsuario
    r3 = bot_isolado.responder("Quero saber mais da opção 1", token)
    assert r3.estado_atual == "EXIBINDO_OFERTAS"
    assert "https://www.cheapshark.com/redirect?dealID=" in r3.resposta

    # Turno 4: Cria alerta para o jogo em contexto sem repetir o nome do jogo
    r4 = bot_isolado.responder("Me avisa quando baixar para 30 reais", token)
    assert r4.intencao == "criar_alerta"
    assert "Hollow Knight" in r4.resposta
    assert "R$ 30,00" in r4.resposta


def test_cenario_1_pos_venda_pedido_com_validacao_cpf_e_lgpd(
    bot_isolado: DealHuntersBot,
) -> None:
    token, _ = bot_isolado.iniciar_sessao()

    # Turno 1: Consulta pedido pré-populado DH-1001
    r1 = bot_isolado.responder("Quero consultar meu pedido DH-1001", token)
    assert r1.estado_atual == "AGUARDANDO_CPF_VALIDACAO"
    assert "CPF" in r1.resposta

    # Turno 2: Informa CPF correto -> libera chave e mascara CPF no ConversaLog
    r2 = bot_isolado.responder("111.222.333-44", token)
    assert r2.estado_atual == "INICIO"
    assert "DHW3-8821-STEAM-9941" in r2.resposta
    assert "***.222.333-**" in r2.resposta

    # Verifica auditoria LGPD na tabela ConversaLog: CPF puro NÃO pode estar gravado
    logs = bot_isolado.db.listar_logs_sessao(token, limite=20)
    todos_textos = " ".join(log.mensagem_sanitizada for log in logs)
    assert "111.222.333-44" not in todos_textos
    assert "***.222.333-**" in todos_textos


def test_cenario_2_girias_e_erros_ortograficos(bot_isolado: DealHuntersBot) -> None:
    token, _ = bot_isolado.iniciar_sessao()

    # Apelido + abreviações ("qnt ta o bruxeiro 3?")
    r1 = bot_isolado.responder("qnt ta o bruxeiro 3?", token)
    assert r1.intencao == "consultar_preco"
    assert "The Witcher 3" in r1.resposta

    # Gírias + erro ortográfico + orçamento + loja
    r2 = bot_isolado.responder("tem promo massa ate 60 conto na steam?", token)
    assert r2.intencao == "buscar_promocoes"
    assert r2.debug_info["entidades"].get("loja") == "Steam"
    assert r2.debug_info["entidades"].get("valor_brl") == 60.0

    # Erro ortográfico em consulta de pedido
    r3 = bot_isolado.responder("kero ver meu pedidu DH-1002", token)
    assert r3.intencao == "consultar_pedido"
    assert r3.estado_atual == "AGUARDANDO_CPF_VALIDACAO"


def test_cenario_3_fallback_duas_falhas_consecutivas_transbordo(
    bot_isolado: DealHuntersBot,
) -> None:
    token, _ = bot_isolado.iniciar_sessao()

    # 1ª falha: pede reformulação sem transferir ainda
    r1 = bot_isolado.responder("qual a receita de bolo de cenoura?", token)
    assert r1.intencao == "desconhecida"
    assert r1.falhas_consecutivas == 1
    assert r1.transbordo_ativo is False

    # 2ª falha consecutiva: aciona transbordo humano e grava em TransbordoHumano
    r2 = bot_isolado.responder("quem ganhou a partida de tenis ontem?", token)
    assert r2.estado_atual == "TRANSBORDO_HUMANO"
    assert r2.falhas_consecutivas == 2
    assert r2.transbordo_ativo is True
    assert "DH-ATD-" in r2.resposta

    sessao = bot_isolado.db.obter_ou_criar_sessao(session_token=token)
    assert sessao.falhas_consecutivas == 2

    transbordos = bot_isolado.db.listar_transbordos(limite=5)
    assert len(transbordos) >= 1
    assert transbordos[0].motivo == "FALLBACK_CONSECUTIVO"


def test_cenario_3_transbordo_imediato_por_insatisfacao_urgencia(
    bot_isolado: DealHuntersBot,
) -> None:
    token, _ = bot_isolado.iniciar_sessao()

    r = bot_isolado.responder(
        "Que absurdo! Paguei e a chave deu inválida, quero reembolso urgente!",
        token,
    )
    assert r.sentimento == "CRITICO_URGENTE"
    assert r.estado_atual == "TRANSBORDO_HUMANO"
    assert r.transbordo_ativo is True
    assert "ALTA" in r.resposta

    transbordos = bot_isolado.db.listar_transbordos(limite=5)
    assert transbordos[0].motivo == "INSATISFACAO_URGENCIA"
    assert transbordos[0].prioridade == "ALTA"


def test_resiliencia_api_requests_timeout_aciona_fallback_gracioso() -> None:
    client = CheapSharkClient(offline_mode=False, max_retries=1)
    with patch.object(
        client._session, "get", side_effect=requests.Timeout("Timeout simulado")
    ):
        resultado = client.consultar_preco_jogo("Hollow Knight")

    assert resultado.sucesso is True
    assert resultado.origem == "cache_fallback"
    assert resultado.erro_amigavel is not None
    assert len(resultado.dados["ofertas"]) >= 1


def test_direito_anonimizacao_lgpd(bot_isolado: DealHuntersBot) -> None:
    token, _ = bot_isolado.iniciar_sessao()
    bot_isolado.responder("Quanto custa Elden Ring?", token)

    r_anon = bot_isolado.responder("Quero apagar meus dados da conversa", token)
    assert r_anon.intencao == "anonimizar_lgpd"
    assert "anonimizados" in r_anon.resposta.lower()

    logs = bot_isolado.db.listar_logs_sessao(token, limite=20)
    assert all(log.anonimizado_lgpd for log in logs)


def test_itad_preco_real_brl_sem_conversao_usd(tmp_path: Path) -> None:
    """Valida que IsThereAnyDealClient retorna preços reais em BRL (brl_nativo) sem exibir US$."""
    client = IsThereAnyDealClient(api_key="chave-teste", offline_mode=False)

    mock_search = MagicMock()
    mock_search.status_code = 200
    mock_search.raise_for_status.return_value = None
    mock_search.json.return_value = [
        {"id": "uuid-hk", "title": "Hollow Knight", "type": "game"}
    ]

    mock_prices = MagicMock()
    mock_prices.status_code = 200
    mock_prices.raise_for_status.return_value = None
    mock_prices.json.return_value = [
        {
            "id": "uuid-hk",
            "historyLow": {"all": {"amount": 9.49, "currency": "BRL"}},
            "deals": [
                {
                    "shop": {"id": 61, "name": "Steam"},
                    "price": {"amount": 46.99, "currency": "BRL"},
                    "regular": {"amount": 46.99, "currency": "BRL"},
                    "cut": 0,
                    "url": "https://itad.link/steam-hk",
                },
                {
                    "shop": {"id": 50, "name": "Nuuvem"},
                    "price": {"amount": 39.99, "currency": "BRL"},
                    "regular": {"amount": 46.99, "currency": "BRL"},
                    "cut": 15,
                    "url": "https://itad.link/nuuvem-hk",
                },
            ],
        }
    ]

    db = Database(database_url=f"sqlite:///{(tmp_path / 'itad.db').as_posix()}")
    bot = DealHuntersBot(database=db, api_client=client, seed_on_start=False)
    token, _ = bot.iniciar_sessao()

    with (
        patch.object(client._session, "get", return_value=mock_search),
        patch.object(client._session, "post", return_value=mock_prices),
    ):
        resp = bot.responder("Quanto custa Hollow Knight?", token)

    assert resp.origem_dados == "api_live"
    assert "R$ 39,99" in resp.resposta
    assert "R$ 46,99" in resp.resposta
    assert "Nuuvem" in resp.resposta
    assert "IsThereAnyDeal API (preço real em BRL — Brasil)" in resp.resposta
    assert "US$" not in resp.resposta


def test_itad_timeout_aciona_fallback_gracioso() -> None:
    client = IsThereAnyDealClient(
        api_key="chave-teste", offline_mode=False, max_retries=1
    )
    with patch.object(
        client._session, "get", side_effect=requests.Timeout("Timeout ITAD")
    ):
        resultado = client.consultar_preco_jogo("Hollow Knight")

    assert resultado.sucesso is True
    assert resultado.origem == "cache_fallback"
    assert resultado.erro_amigavel is not None
    assert "IsThereAnyDeal" in resultado.erro_amigavel
    assert len(resultado.dados["ofertas"]) >= 1
