"""Clientes HTTP para busca de preços (IsThereAnyDeal e CheapShark)."""

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
    origem: str = "api_live"
    cotacao_usd_brl: Optional[float] = config.DEFAULT_USD_BRL_RATE
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


ITAD_DEFAULT_SHOPS_MAP: Dict[int, str] = {
    4: "Blizzard",
    6: "Fanatical",
    15: "DreamGame",
    16: "Epic Games Store",
    24: "GamersGate",
    26: "GamesPlanet UK",
    27: "GamesPlanet DE",
    28: "GamesPlanet FR",
    29: "GamesPlanet US",
    35: "GOG",
    36: "GreenManGaming",
    37: "Humble Store",
    50: "Nuuvem",
    52: "EA Store",
    61: "Steam",
    62: "Ubisoft Store",
}

ITAD_STORE_NAME_ALIASES: Dict[str, List[int]] = {
    "steam": [61],
    "nuuvem": [50],
    "gog": [35],
    "epic": [16],
    "epic games": [16],
    "epic games store": [16],
    "epic game store": [16],
    "humble": [37],
    "humble store": [37],
    "humble bundle": [37],
    "fanatical": [6],
    "greenmangaming": [36],
    "green man gaming": [36],
    "gmg": [36],
    "gamersgate": [24],
    "ubisoft": [62],
    "ubisoft connect": [62],
    "ubisoft store": [62],
    "uplay": [62],
    "origin": [52],
    "ea": [52],
    "ea store": [52],
    "origin / ea": [52],
    "blizzard": [4],
    "battle.net": [4],
    "dreamgame": [15],
    "gamesplanet": [26, 27, 28, 29],
}


class IsThereAnyDealClient:
    """
    Cliente REST para a API IsThereAnyDeal (v2/v3).
    Retorna preços REAIS regionalizados em BRL (country=BR) sem conversão artificial.
    Inclui fallback automático para o catálogo local em caso de queda de rede ou modo offline.
    """

    def __init__(
        self,
        api_key: str = config.ITAD_API_KEY,
        base_url: str = config.ITAD_BASE_URL,
        country: str = config.ITAD_COUNTRY,
        timeout: float = config.API_TIMEOUT_SECONDS,
        max_retries: int = config.API_MAX_RETRIES,
        offline_mode: bool = config.OFFLINE_MODE,
        fallback_catalog_path: Path = config.FALLBACK_CATALOG_PATH,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.country = (country or "BR").strip().upper()
        self.timeout = timeout
        self.max_retries = max_retries
        self.offline_mode = offline_mode
        self.fallback_catalog_path = Path(fallback_catalog_path)

        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": config.CHEAPSHARK_USER_AGENT,
                "Accept": "application/json",
            }
        )
        self._shops_cache: Dict[int, str] = dict(ITAD_DEFAULT_SHOPS_MAP)
        self._shops_loaded_live: bool = False
        self._fallback_helper = CheapSharkClient(
            offline_mode=True,
            fallback_catalog_path=self.fallback_catalog_path,
        )

    @staticmethod
    def resolver_shop_ids(nome_loja: Optional[str]) -> Optional[List[int]]:
        if not nome_loja:
            return None
        chave = nome_loja.strip().lower()
        return ITAD_STORE_NAME_ALIASES.get(chave)

    def _request_json(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Any] = None,
    ) -> Tuple[Optional[Any], Optional[str]]:
        if self.offline_mode:
            return None, "Modo offline ativado por configuração."
        if not self.api_key:
            return None, "ITAD_API_KEY não configurada no ambiente (.env)."

        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        req_params = dict(params or {})
        req_params["key"] = self.api_key
        ultimo_erro: Optional[str] = None

        for tentativa in range(1, self.max_retries + 1):
            try:
                if method.upper() == "POST":
                    resp = self._session.post(
                        url,
                        params=req_params,
                        json=json_body,
                        timeout=self.timeout,
                    )
                else:
                    resp = self._session.get(
                        url,
                        params=req_params,
                        timeout=self.timeout,
                    )

                if resp.status_code == 429:
                    ultimo_erro = "Rate limit da IsThereAnyDeal atingido (HTTP 429)."
                    if tentativa < self.max_retries:
                        time.sleep(0.8 * tentativa)
                        continue
                    return None, ultimo_erro

                resp.raise_for_status()
                return resp.json(), None

            except requests.Timeout:
                ultimo_erro = f"Timeout após {self.timeout}s ao consultar {endpoint}."
            except requests.ConnectionError:
                ultimo_erro = "Falha de conexão de rede com a API IsThereAnyDeal."
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else "?"
                ultimo_erro = f"Erro HTTP {status} retornado pela API IsThereAnyDeal."
                if isinstance(status, int) and 400 <= status < 500:
                    break
            except ValueError:
                ultimo_erro = "Resposta JSON inválida recebida da API IsThereAnyDeal."
                break

            if tentativa < self.max_retries:
                time.sleep(0.4 * tentativa)

        return None, ultimo_erro

    def carregar_lojas(self) -> Dict[int, str]:
        if self._shops_loaded_live or self.offline_mode or not self.api_key:
            return self._shops_cache

        payload, erro = self._request_json(
            "GET", "/service/shops/v1", params={"country": self.country}
        )
        if payload and isinstance(payload, list) and not erro:
            for loja in payload:
                sid = loja.get("id")
                title = loja.get("title")
                if isinstance(sid, int) and title:
                    self._shops_cache[sid] = str(title)
            self._shops_loaded_live = True
        return self._shops_cache

    @staticmethod
    def _normalizar_oferta_brl(
        *,
        titulo: str,
        deal_id: str,
        store_name: str,
        store_id: str,
        sale_price_brl: float,
        normal_price_brl: float,
        savings_pct: float,
        link: str,
        cheapest_ever_brl: Optional[float] = None,
    ) -> Dict[str, Any]:
        return {
            "titulo": titulo,
            "deal_id": deal_id,
            "loja": store_name,
            "store_id": store_id,
            "preco_usd": None,
            "preco_normal_usd": None,
            "preco_brl": round(sale_price_brl, 2),
            "preco_normal_brl": round(normal_price_brl, 2),
            "desconto_pct": round(savings_pct, 1),
            "metacritic": None,
            "menor_preco_historico_usd": None,
            "menor_preco_historico_brl": (
                round(cheapest_ever_brl, 2) if cheapest_ever_brl is not None else None
            ),
            "link": link,
        }

    @staticmethod
    def _escolher_melhor_jogo(
        candidatos: List[Dict[str, Any]], titulo: str
    ) -> Dict[str, Any]:
        titulo_lower = titulo.strip().lower()
        for c in candidatos:
            if (
                str(c.get("title", "")).strip().lower() == titulo_lower
                and c.get("type") == "game"
            ):
                return c
        for c in candidatos:
            if str(c.get("title", "")).strip().lower() == titulo_lower:
                return c
        for c in candidatos:
            if (
                titulo_lower in str(c.get("title", "")).lower()
                and c.get("type") == "game"
            ):
                return c
        for c in candidatos:
            if c.get("type") == "game":
                return c
        return candidatos[0]

    def consultar_preco_jogo(
        self,
        titulo: str,
        loja_filtro: Optional[str] = None,
        limite_ofertas: int = 4,
    ) -> ApiResult:
        """
        Busca o jogo na IsThereAnyDeal (GET /games/search/v1 -> POST /games/prices/v3?country=BR)
        e retorna preços REAIS regionalizados em BRL.
        """
        shop_ids_filtro = self.resolver_shop_ids(loja_filtro)

        search_payload, erro_search = self._request_json(
            "GET", "/games/search/v1", params={"title": titulo, "results": 8}
        )

        if erro_search is None and isinstance(search_payload, list):
            if not search_payload:
                return ApiResult(
                    sucesso=True,
                    dados={"jogo": titulo, "ofertas": []},
                    origem="api_live",
                    cotacao_usd_brl=None,
                    origem_cotacao="brl_nativo",
                )

            melhor_jogo = self._escolher_melhor_jogo(search_payload, titulo)
            game_id = str(melhor_jogo.get("id", ""))
            nome_oficial = str(melhor_jogo.get("title") or titulo)

            price_params: Dict[str, Any] = {
                "country": self.country,
                "vouchers": "true",
            }
            if shop_ids_filtro:
                price_params["shops"] = ",".join(str(sid) for sid in shop_ids_filtro)

            prices_payload, erro_prices = self._request_json(
                "POST",
                "/games/prices/v3",
                params=price_params,
                json_body=[game_id],
            )

            if erro_prices is None and isinstance(prices_payload, list):
                item_jogo: Dict[str, Any] = {}
                for entry in prices_payload:
                    if str(entry.get("id", "")) == game_id:
                        item_jogo = entry
                        break
                if not item_jogo and prices_payload:
                    item_jogo = prices_payload[0]

                history_low = item_jogo.get("historyLow") or {}
                low_all = history_low.get("all") or {}
                cheapest_ever_brl: Optional[float] = None
                if low_all.get("amount") is not None:
                    try:
                        cheapest_ever_brl = float(low_all["amount"])
                    except (TypeError, ValueError):
                        cheapest_ever_brl = None

                ofertas: List[Dict[str, Any]] = []
                for deal in item_jogo.get("deals", []):
                    shop = deal.get("shop") or {}
                    price_obj = deal.get("price") or {}
                    regular_obj = deal.get("regular") or price_obj
                    if price_obj.get("amount") is None:
                        continue
                    try:
                        sale_brl = float(price_obj["amount"])
                        regular_brl = float(regular_obj.get("amount", sale_brl))
                        cut = float(deal.get("cut") or 0.0)
                    except (TypeError, ValueError):
                        continue

                    sid = str(shop.get("id", ""))
                    shop_name = str(shop.get("name") or f"Loja #{sid}")
                    deal_url = str(deal.get("url") or "")

                    ofertas.append(
                        self._normalizar_oferta_brl(
                            titulo=nome_oficial,
                            deal_id=f"itad-{game_id}-{sid}",
                            store_name=shop_name,
                            store_id=sid,
                            sale_price_brl=sale_brl,
                            normal_price_brl=regular_brl,
                            savings_pct=cut,
                            link=deal_url,
                            cheapest_ever_brl=cheapest_ever_brl,
                        )
                    )

                ofertas.sort(key=lambda item: item["preco_brl"])
                return ApiResult(
                    sucesso=True,
                    dados={
                        "jogo": nome_oficial,
                        "game_id": game_id,
                        "menor_preco_historico_usd": None,
                        "menor_preco_historico_brl": (
                            round(cheapest_ever_brl, 2)
                            if cheapest_ever_brl is not None
                            else None
                        ),
                        "ofertas": ofertas[:limite_ofertas],
                    },
                    origem="api_live",
                    cotacao_usd_brl=None,
                    origem_cotacao="brl_nativo",
                )

            erro_final = erro_prices
        else:
            erro_final = erro_search

        fallback_res = self._fallback_helper.consultar_preco_jogo(
            titulo=titulo,
            loja_filtro=loja_filtro,
            limite_ofertas=limite_ofertas,
        )
        fallback_res.detalhe_tecnico = erro_final
        if not self.offline_mode and fallback_res.sucesso:
            fallback_res.erro_amigavel = (
                "Aviso de contingência: a API IsThereAnyDeal está instável no momento; "
                "exibindo preços de referência do cache local."
            )
        return fallback_res

    def buscar_promocoes(
        self,
        preco_max_brl: Optional[float] = None,
        loja_filtro: Optional[str] = None,
        titulo_filtro: Optional[str] = None,
        limite: int = 5,
    ) -> ApiResult:
        """
        Consulta promoções ativas em BRL real via GET /deals/v2?country=BR
        com filtros opcionais de orçamento em reais e loja específica.
        """
        if titulo_filtro:
            res_jogo = self.consultar_preco_jogo(
                titulo=titulo_filtro,
                loja_filtro=loja_filtro,
                limite_ofertas=max(limite, 5),
            )
            if res_jogo.sucesso and res_jogo.origem == "api_live":
                ofertas_filtradas = (res_jogo.dados or {}).get("ofertas", [])
                if preco_max_brl is not None and preco_max_brl > 0:
                    ofertas_filtradas = [
                        o for o in ofertas_filtradas if o["preco_brl"] <= preco_max_brl
                    ]
                return ApiResult(
                    sucesso=True,
                    dados={
                        "ofertas": ofertas_filtradas[:limite],
                        "filtro_preco_max_brl": preco_max_brl,
                        "filtro_loja": loja_filtro,
                    },
                    origem="api_live",
                    cotacao_usd_brl=None,
                    origem_cotacao="brl_nativo",
                )

        shop_ids_filtro = self.resolver_shop_ids(loja_filtro)
        params: Dict[str, Any] = {
            "country": self.country,
            "limit": 60,
            "sort": "-hot",
            "nondeals": "false",
        }
        if shop_ids_filtro:
            params["shops"] = ",".join(str(sid) for sid in shop_ids_filtro)

        payload, erro = self._request_json("GET", "/deals/v2", params=params)
        if erro is None and isinstance(payload, dict):
            itens = payload.get("list") or []
            ofertas_jogos: List[Dict[str, Any]] = []
            ofertas_outros: List[Dict[str, Any]] = []

            for entry in itens:
                deal = entry.get("deal") or {}
                shop = deal.get("shop") or {}
                price_obj = deal.get("price") or {}
                regular_obj = deal.get("regular") or price_obj
                if price_obj.get("amount") is None:
                    continue
                try:
                    sale_brl = float(price_obj["amount"])
                    regular_brl = float(regular_obj.get("amount", sale_brl))
                    cut = float(deal.get("cut") or 0.0)
                except (TypeError, ValueError):
                    continue

                if preco_max_brl is not None and preco_max_brl > 0 and sale_brl > preco_max_brl:
                    continue

                sid = str(shop.get("id", ""))
                shop_name = str(shop.get("name") or f"Loja #{sid}")
                oferta = self._normalizar_oferta_brl(
                    titulo=str(entry.get("title") or "Jogo sem título"),
                    deal_id=f"itad-{entry.get('id', '')}-{sid}",
                    store_name=shop_name,
                    store_id=sid,
                    sale_price_brl=sale_brl,
                    normal_price_brl=regular_brl,
                    savings_pct=cut,
                    link=str(deal.get("url") or ""),
                )
                if entry.get("type") == "game":
                    ofertas_jogos.append(oferta)
                else:
                    ofertas_outros.append(oferta)

            ofertas = (ofertas_jogos + ofertas_outros)[:limite]
            return ApiResult(
                sucesso=True,
                dados={
                    "ofertas": ofertas,
                    "filtro_preco_max_brl": preco_max_brl,
                    "filtro_loja": loja_filtro,
                },
                origem="api_live",
                cotacao_usd_brl=None,
                origem_cotacao="brl_nativo",
            )

        fallback_res = self._fallback_helper.buscar_promocoes(
            preco_max_brl=preco_max_brl,
            loja_filtro=loja_filtro,
            titulo_filtro=titulo_filtro,
            limite=limite,
        )
        fallback_res.detalhe_tecnico = erro
        if not self.offline_mode and fallback_res.sucesso:
            fallback_res.erro_amigavel = (
                "Aviso de contingência: a API IsThereAnyDeal está instável no momento; "
                "exibindo promoções de referência do cache local."
            )
        return fallback_res


def criar_api_client(offline_mode: bool = config.OFFLINE_MODE) -> Any:
    """Fábrica padrão: usa IsThereAnyDeal (BRL real) quando ITAD_API_KEY estiver configurada."""
    if config.ITAD_API_KEY:
        return IsThereAnyDealClient(offline_mode=offline_mode)
    return CheapSharkClient(offline_mode=offline_mode)
