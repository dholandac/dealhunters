"""
Eixo 2 — Integração Sistêmica via API REST (requests).

Consome em tempo real:
1. CheapShark API v1.0 (https://www.cheapshark.com/api/1.0) — gratuita e sem API key:
   - GET /stores : catálogo de lojas digitais
   - GET /games?title=... : busca de jogos por título
   - GET /games?id=...    : detalhes e ofertas de um jogo em múltiplas lojas + menor preço histórico
   - GET /deals  : promoções filtradas por orçamento, loja e desconto
2. AwesomeAPI (https://economia.awesomeapi.com.br/json/last/USD-BRL) — cotação USD->BRL em tempo real.

Inclui tratamento robusto de exceções (Timeout, ConnectionError, HTTPError, JSON inválido),
retentativas com backoff leve e fallback automático para catálogo local em caso de queda de rede.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

import config


DEFAULT_STORES_MAP: Dict[str, str] = {
    "1": "Steam",
    "2": "GamersGate",
    "3": "GreenManGaming",
    "7": "GOG",
    "8": "Origin / EA",
    "11": "Humble Store",
    "13": "Ubisoft Connect",
    "15": "Fanatical",
    "25": "Epic Games Store",
    "27": "Gamesplanet",
    "31": "Blizzard",
    "35": "DreamGame",
}

STORE_NAME_ALIASES: Dict[str, str] = {
    "steam": "1",
    "gamersgate": "2",
    "greenmangaming": "3",
    "green man gaming": "3",
    "gmg": "3",
    "gog": "7",
    "origin": "8",
    "ea": "8",
    "humble": "11",
    "humble store": "11",
    "humble bundle": "11",
    "ubisoft": "13",
    "uplay": "13",
    "fanatical": "15",
    "epic": "25",
    "epic games": "25",
    "epic games store": "25",
    "gamesplanet": "27",
    "blizzard": "31",
    "battle.net": "31",
}


@dataclass
class ApiResult:
    """Envelope padronizado de resposta da camada de integração REST."""

    sucesso: bool
    dados: Any = None
    origem: str = "api_live"  # api_live | cache_fallback | erro
    cotacao_usd_brl: float = config.DEFAULT_USD_BRL_RATE
    origem_cotacao: str = "fallback"
    erro_amigavel: Optional[str] = None
    detalhe_tecnico: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExchangeRateClient:
    """Consulta a cotação USD -> BRL em tempo real com cache e fallback seguro."""

    def __init__(
        self,
        url: str = config.EXCHANGE_RATE_URL,
        fallback_rate: float = config.DEFAULT_USD_BRL_RATE,
        timeout: float = config.API_TIMEOUT_SECONDS,
        cache_ttl: int = config.EXCHANGE_CACHE_TTL_SECONDS,
        offline_mode: bool = config.OFFLINE_MODE,
    ) -> None:
        self.url = url
        self.fallback_rate = fallback_rate
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.offline_mode = offline_mode
        self._cached_rate: Optional[float] = None
        self._cached_at: float = 0.0
        self._cached_source: str = "fallback"
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": config.CHEAPSHARK_USER_AGENT})

    def obter_taxa_usd_brl(self) -> Tuple[float, str]:
        """Retorna (taxa_usd_brl, origem) usando cache de curto prazo e resiliência."""
        if self.offline_mode:
            return self.fallback_rate, "fallback_offline"

        agora = time.monotonic()
        if (
            self._cached_rate is not None
            and (agora - self._cached_at) < self.cache_ttl
        ):
            return self._cached_rate, self._cached_source

        try:
            resp = self._session.get(self.url, timeout=min(self.timeout, 5.0))
            resp.raise_for_status()
            payload = resp.json()
            bid_str = payload.get("USDBRL", {}).get("bid")
            taxa = float(bid_str)
            if taxa > 0:
                self._cached_rate = round(taxa, 4)
                self._cached_at = agora
                self._cached_source = "awesomeapi_live"
                return self._cached_rate, self._cached_source
        except (
            requests.RequestException,
            ValueError,
            TypeError,
            KeyError,
        ):
            pass

        return self.fallback_rate, "fallback_padrao"


class CheapSharkClient:
    """
    Cliente REST desacoplado para a CheapShark API v1.0.
    Todas as chamadas retornam preços convertidos para BRL e o link oficial de redirect.
    """

    def __init__(
        self,
        base_url: str = config.CHEAPSHARK_BASE_URL,
        timeout: float = config.API_TIMEOUT_SECONDS,
        max_retries: int = config.API_MAX_RETRIES,
        offline_mode: bool = config.OFFLINE_MODE,
        fallback_catalog_path: Path = config.FALLBACK_CATALOG_PATH,
        exchange_client: Optional[ExchangeRateClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self.fallback_catalog_path = Path(fallback_catalog_path)
        self.exchange_client = exchange_client or ExchangeRateClient(
            offline_mode=offline_mode
        )

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": config.CHEAPSHARK_USER_AGENT,
                "Accept": "application/json",
            }
        )
        self._stores_cache: Dict[str, str] = dict(DEFAULT_STORES_MAP)
        self._stores_loaded_live: bool = False
        self._fallback_catalog: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Utilitários internos de HTTP e Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def montar_link_oferta(deal_id: str) -> str:
        """Usa obrigatoriamente o redirect oficial da CheapShark conforme política da API."""
        return config.CHEAPSHARK_REDIRECT_TEMPLATE.format(deal_id=deal_id)

    @staticmethod
    def resolver_store_id(nome_loja: Optional[str]) -> Optional[str]:
        if not nome_loja:
            return None
        chave = nome_loja.strip().lower()
        return STORE_NAME_ALIASES.get(chave)

    def _carregar_fallback_catalog(self) -> Dict[str, Any]:
        if self._fallback_catalog is not None:
            return self._fallback_catalog
        try:
            with self.fallback_catalog_path.open("r", encoding="utf-8") as fp:
                self._fallback_catalog = json.load(fp)
        except (OSError, json.JSONDecodeError):
            self._fallback_catalog = {"stores": DEFAULT_STORES_MAP, "games": []}
        return self._fallback_catalog

    def _request_json(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Executa requisição GET com timeout, tratamento de HTTP 429/5xx e retentativas.
        Retorna (payload_json, mensagem_erro_tecnico).
        """
        if self.offline_mode:
            return None, "Modo offline ativado por configuração."

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        ultimo_erro: Optional[str] = None

        for tentativa in range(1, self.max_retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    ultimo_erro = "Rate limit da CheapShark atingido (HTTP 429)."
                    if tentativa < self.max_retries:
                        time.sleep(0.8 * tentativa)
                        continue
                    return None, ultimo_erro

                resp.raise_for_status()
                return resp.json(), None

            except requests.Timeout:
                ultimo_erro = f"Timeout após {self.timeout}s ao consultar {endpoint}."
            except requests.ConnectionError:
                ultimo_erro = "Falha de conexão de rede com a API CheapShark."
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                ultimo_erro = f"Erro HTTP {status} retornado pela API CheapShark."
                if isinstance(status, int) and 400 <= status < 500:
                    break
            except ValueError:
                ultimo_erro = "Resposta JSON inválida recebida da API CheapShark."
                break

            if tentativa < self.max_retries:
                time.sleep(0.4 * tentativa)

        return None, ultimo_erro

    def carregar_lojas(self) -> Dict[str, str]:
        """Consulta GET /stores uma vez por execução para manter nomes de lojas atualizados."""
        if self._stores_loaded_live or self.offline_mode:
            return self._stores_cache

        payload, erro = self._request_json("/stores")
        if payload and isinstance(payload, list) and not erro:
            for loja in payload:
                store_id = str(loja.get("storeID", ""))
                store_name = loja.get("storeName")
                if store_id and store_name:
                    self._stores_cache[store_id] = store_name
            self._stores_loaded_live = True
        return self._stores_cache

    def _nome_loja(self, store_id: Any) -> str:
        lojas = self.carregar_lojas()
        return lojas.get(str(store_id), f"Loja #{store_id}")

    @staticmethod
    def _normalizar_oferta(
        *,
        titulo: str,
        deal_id: str,
        store_name: str,
        store_id: str,
        sale_price_usd: float,
        normal_price_usd: float,
        savings_pct: float,
        taxa_usd_brl: float,
        metacritic: Optional[str] = None,
        cheapest_ever_usd: Optional[float] = None,
    ) -> Dict[str, Any]:
        sale_brl = round(sale_price_usd * taxa_usd_brl, 2)
        normal_brl = round(normal_price_usd * taxa_usd_brl, 2)
        cheapest_brl = (
            round(cheapest_ever_usd * taxa_usd_brl, 2)
            if cheapest_ever_usd is not None
            else None
        )
        return {
            "titulo": titulo,
            "deal_id": deal_id,
            "loja": store_name,
            "store_id": store_id,
            "preco_usd": round(sale_price_usd, 2),
            "preco_normal_usd": round(normal_price_usd, 2),
            "preco_brl": sale_brl,
            "preco_normal_brl": normal_brl,
            "desconto_pct": round(savings_pct, 1),
            "metacritic": metacritic if metacritic not in (None, "", "0") else None,
            "menor_preco_historico_usd": cheapest_ever_usd,
            "menor_preco_historico_brl": cheapest_brl,
            "link": CheapSharkClient.montar_link_oferta(deal_id),
        }

    # ------------------------------------------------------------------
    # Operações Públicas de Negócio
    # ------------------------------------------------------------------

    def consultar_preco_jogo(
        self,
        titulo: str,
        loja_filtro: Optional[str] = None,
        limite_ofertas: int = 4,
    ) -> ApiResult:
        """
        Busca um jogo pelo título na CheapShark (GET /games -> GET /game?id=...)
        e retorna as melhores ofertas em múltiplas lojas com conversão USD -> BRL.
        """
        taxa, origem_taxa = self.exchange_client.obter_taxa_usd_brl()
        store_id_filtro = self.resolver_store_id(loja_filtro)

        # 1) Busca o jogo principal na API live
        jogos_payload, erro_games = self._request_json(
            "/games", params={"title": titulo, "limit": 5}
        )

        if erro_games is None and isinstance(jogos_payload, list):
            if not jogos_payload:
                return ApiResult(
                    sucesso=True,
                    dados={"jogo": titulo, "ofertas": []},
                    origem="api_live",
                    cotacao_usd_brl=taxa,
                    origem_cotacao=origem_taxa,
                )

            # Prioriza correspondência mais próxima do título solicitado
            titulo_lower = titulo.strip().lower()
            melhor_jogo = jogos_payload[0]
            for candidato in jogos_payload:
                nome_cand = str(candidato.get("external", "")).lower()
                if nome_cand == titulo_lower or titulo_lower in nome_cand:
                    melhor_jogo = candidato
                    break

            game_id = str(melhor_jogo.get("gameID", ""))
            nome_oficial = str(melhor_jogo.get("external") or titulo)

            detalhes_payload, erro_det = self._request_json(
                "/games", params={"id": game_id}
            )
            ofertas: List[Dict[str, Any]] = []
            cheapest_ever: Optional[float] = None

            if erro_det is None and isinstance(detalhes_payload, dict):
                info = detalhes_payload.get("info", {})
                nome_oficial = str(info.get("title") or nome_oficial)
                cheapest_obj = detalhes_payload.get("cheapestPriceEver") or {}
                if cheapest_obj.get("price") is not None:
                    try:
                        cheapest_ever = float(cheapest_obj["price"])
                    except (TypeError, ValueError):
                        cheapest_ever = None

                for deal in detalhes_payload.get("deals", []):
                    sid = str(deal.get("storeID", ""))
                    if store_id_filtro and sid != store_id_filtro:
                        continue
                    try:
                        sale_usd = float(deal.get("price", 0))
                        retail_usd = float(deal.get("retailPrice", sale_usd))
                        savings = float(deal.get("savings", 0))
                    except (TypeError, ValueError):
                        continue

                    ofertas.append(
                        self._normalizar_oferta(
                            titulo=nome_oficial,
                            deal_id=str(deal.get("dealID", "")),
                            store_name=self._nome_loja(sid),
                            store_id=sid,
                            sale_price_usd=sale_usd,
                            normal_price_usd=retail_usd,
                            savings_pct=savings,
                            taxa_usd_brl=taxa,
                            cheapest_ever_usd=cheapest_ever,
                        )
                    )
            else:
                # Fallback leve usando dados do próprio /games caso /game falhe
                try:
                    cheapest_usd = float(melhor_jogo.get("cheapest", 0))
                except (TypeError, ValueError):
                    cheapest_usd = 0.0
                deal_id = str(melhor_jogo.get("cheapestDealID", ""))
                if deal_id:
                    ofertas.append(
                        self._normalizar_oferta(
                            titulo=nome_oficial,
                            deal_id=deal_id,
                            store_name="Melhor Loja (CheapShark)",
                            store_id="",
                            sale_price_usd=cheapest_usd,
                            normal_price_usd=cheapest_usd,
                            savings_pct=0.0,
                            taxa_usd_brl=taxa,
                        )
                    )

            ofertas.sort(key=lambda item: item["preco_usd"])
            return ApiResult(
                sucesso=True,
                dados={
                    "jogo": nome_oficial,
                    "game_id": game_id,
                    "menor_preco_historico_usd": cheapest_ever,
                    "menor_preco_historico_brl": (
                        round(cheapest_ever * taxa, 2)
                        if cheapest_ever is not None
                        else None
                    ),
                    "ofertas": ofertas[:limite_ofertas],
                },
                origem="api_live",
                cotacao_usd_brl=taxa,
                origem_cotacao=origem_taxa,
            )

        # 2) Contingência automática (catálogo local) em caso de falha de rede/API
        return self._consultar_preco_fallback(
            titulo=titulo,
            loja_filtro=store_id_filtro,
            limite_ofertas=limite_ofertas,
            taxa=taxa,
            origem_taxa=origem_taxa,
            detalhe_erro=erro_games,
        )

    def buscar_promocoes(
        self,
        preco_max_brl: Optional[float] = None,
        loja_filtro: Optional[str] = None,
        titulo_filtro: Optional[str] = None,
        limite: int = 5,
    ) -> ApiResult:
        """
        Consulta promoções ativas via GET /deals com filtros opcionais de orçamento (BRL -> USD)
        e loja específica.
        """
        taxa, origem_taxa = self.exchange_client.obter_taxa_usd_brl()
        store_id_filtro = self.resolver_store_id(loja_filtro)

        params: Dict[str, Any] = {
            "onSale": 1,
            "sortBy": "Deal Rating",
            "pageSize": max(limite, 5),
        }
        if preco_max_brl is not None and preco_max_brl > 0:
            params["upperPrice"] = round(preco_max_brl / taxa, 2)
        if store_id_filtro:
            params["storeID"] = store_id_filtro
        if titulo_filtro:
            params["title"] = titulo_filtro

        payload, erro = self._request_json("/deals", params=params)
        if erro is None and isinstance(payload, list):
            ofertas: List[Dict[str, Any]] = []
            for deal in payload[:limite]:
                sid = str(deal.get("storeID", ""))
                try:
                    sale_usd = float(deal.get("salePrice", 0))
                    normal_usd = float(deal.get("normalPrice", sale_usd))
                    savings = float(deal.get("savings", 0))
                except (TypeError, ValueError):
                    continue

                ofertas.append(
                    self._normalizar_oferta(
                        titulo=str(deal.get("title", "Jogo sem título")),
                        deal_id=str(deal.get("dealID", "")),
                        store_name=self._nome_loja(sid),
                        store_id=sid,
                        sale_price_usd=sale_usd,
                        normal_price_usd=normal_usd,
                        savings_pct=savings,
                        taxa_usd_brl=taxa,
                        metacritic=str(deal.get("metacriticScore", "")),
                    )
                )

            return ApiResult(
                sucesso=True,
                dados={
                    "ofertas": ofertas,
                    "filtro_preco_max_brl": preco_max_brl,
                    "filtro_loja": loja_filtro,
                },
                origem="api_live",
                cotacao_usd_brl=taxa,
                origem_cotacao=origem_taxa,
            )

        return self._buscar_promocoes_fallback(
            preco_max_brl=preco_max_brl,
            store_id_filtro=store_id_filtro,
            loja_filtro=loja_filtro,
            titulo_filtro=titulo_filtro,
            limite=limite,
            taxa=taxa,
            origem_taxa=origem_taxa,
            detalhe_erro=erro,
        )

    # ------------------------------------------------------------------
    # Contingência local (Resiliência contra quedas de rede)
    # ------------------------------------------------------------------

    def _consultar_preco_fallback(
        self,
        *,
        titulo: str,
        loja_filtro: Optional[str],
        limite_ofertas: int,
        taxa: float,
        origem_taxa: str,
        detalhe_erro: Optional[str],
    ) -> ApiResult:
        catalog = self._carregar_fallback_catalog()
        stores_map = catalog.get("stores", DEFAULT_STORES_MAP)
        termo = titulo.strip().lower()

        match_game: Optional[Dict[str, Any]] = None
        for game in catalog.get("games", []):
            game_title = str(game.get("title", "")).lower()
            if termo in game_title or game_title in termo:
                match_game = game
                break

        if not match_game:
            # Tenta correspondência por palavras em comum
            palavras_termo = {p for p in termo.split() if len(p) >= 3}
            for game in catalog.get("games", []):
                palavras_game = set(str(game.get("title", "")).lower().split())
                if palavras_termo and (palavras_termo & palavras_game):
                    match_game = game
                    break

        if not match_game:
            return ApiResult(
                sucesso=False,
                dados={"jogo": titulo, "ofertas": []},
                origem="cache_fallback",
                cotacao_usd_brl=taxa,
                origem_cotacao=origem_taxa,
                erro_amigavel=(
                    "A API externa CheapShark está temporariamente indisponível e "
                    f"não encontrei '{titulo}' no catálogo local de contingência."
                ),
                detalhe_tecnico=detalhe_erro,
            )

        ofertas: List[Dict[str, Any]] = []
        cheapest_ever = match_game.get("cheapest_ever_usd")
        for deal in match_game.get("deals", []):
            sid = str(deal.get("storeID", ""))
            if loja_filtro and sid != loja_filtro:
                continue
            ofertas.append(
                self._normalizar_oferta(
                    titulo=str(match_game["title"]),
                    deal_id=str(deal["dealID"]),
                    store_name=stores_map.get(sid, f"Loja #{sid}"),
                    store_id=sid,
                    sale_price_usd=float(deal["salePrice"]),
                    normal_price_usd=float(deal["normalPrice"]),
                    savings_pct=float(deal["savings"]),
                    taxa_usd_brl=taxa,
                    metacritic=str(deal.get("metacriticScore", "")),
                    cheapest_ever_usd=float(cheapest_ever)
                    if cheapest_ever is not None
                    else None,
                )
            )

        ofertas.sort(key=lambda item: item["preco_usd"])
        return ApiResult(
            sucesso=True,
            dados={
                "jogo": str(match_game["title"]),
                "game_id": str(match_game.get("gameID", "")),
                "menor_preco_historico_usd": cheapest_ever,
                "menor_preco_historico_brl": (
                    round(float(cheapest_ever) * taxa, 2)
                    if cheapest_ever is not None
                    else None
                ),
                "ofertas": ofertas[:limite_ofertas],
            },
            origem="cache_fallback",
            cotacao_usd_brl=taxa,
            origem_cotacao=origem_taxa,
            erro_amigavel=(
                "Aviso de contingência: a API CheapShark está instável no momento; "
                "exibindo preços de referência do cache local."
            ),
            detalhe_tecnico=detalhe_erro,
        )

    def _buscar_promocoes_fallback(
        self,
        *,
        preco_max_brl: Optional[float],
        store_id_filtro: Optional[str],
        loja_filtro: Optional[str],
        titulo_filtro: Optional[str],
        limite: int,
        taxa: float,
        origem_taxa: str,
        detalhe_erro: Optional[str],
    ) -> ApiResult:
        catalog = self._carregar_fallback_catalog()
        stores_map = catalog.get("stores", DEFAULT_STORES_MAP)
        ofertas: List[Dict[str, Any]] = []

        for game in catalog.get("games", []):
            game_title = str(game.get("title", ""))
            if titulo_filtro and titulo_filtro.lower() not in game_title.lower():
                continue
            for deal in game.get("deals", []):
                sid = str(deal.get("storeID", ""))
                if store_id_filtro and sid != store_id_filtro:
                    continue
                sale_usd = float(deal["salePrice"])
                sale_brl = round(sale_usd * taxa, 2)
                if preco_max_brl is not None and sale_brl > preco_max_brl:
                    continue
                ofertas.append(
                    self._normalizar_oferta(
                        titulo=game_title,
                        deal_id=str(deal["dealID"]),
                        store_name=stores_map.get(sid, f"Loja #{sid}"),
                        store_id=sid,
                        sale_price_usd=sale_usd,
                        normal_price_usd=float(deal["normalPrice"]),
                        savings_pct=float(deal["savings"]),
                        taxa_usd_brl=taxa,
                        metacritic=str(deal.get("metacriticScore", "")),
                    )
                )

        ofertas.sort(key=lambda item: item["desconto_pct"], reverse=True)
        return ApiResult(
            sucesso=True,
            dados={
                "ofertas": ofertas[:limite],
                "filtro_preco_max_brl": preco_max_brl,
                "filtro_loja": loja_filtro,
            },
            origem="cache_fallback",
            cotacao_usd_brl=taxa,
            origem_cotacao=origem_taxa,
            erro_amigavel=(
                "Aviso de contingência: a API CheapShark está instável no momento; "
                "exibindo promoções de referência do cache local."
            ),
            detalhe_tecnico=detalhe_erro,
        )
