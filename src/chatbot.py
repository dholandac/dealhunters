"""
Fachada principal `DealHuntersBot` que integra os módulos desacoplados:
NLU -> BusinessLogic -> (CheapSharkClient + Database) -> NLG.

Pode ser consumida tanto pelo terminal (`main.py`) quanto por uma API/Web (`web_app.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.api_client import CheapSharkClient, IsThereAnyDealClient, criar_api_client
from src.business_logic import BusinessLogic
from src.database import Database, sanitizar_entidades_lgpd
from src.nlg import NLG
from src.nlu import NLU


@dataclass
class BotTurnResponse:
    """Resposta completa de um turno do chatbot, incluindo telemetria técnica para /debug."""

    session_token: str
    resposta: str
    estado_anterior: str
    estado_atual: str
    intencao: str
    confianca: float
    sentimento: str
    score_insatisfacao: float
    falhas_consecutivas: int
    transbordo_ativo: bool
    origem_dados: str
    debug_info: Dict[str, Any] = field(default_factory=dict)


class DealHuntersBot:
    """Fachada modular do assistente corporativo Deal Hunters."""

    def __init__(
        self,
        database: Optional[Database] = None,
        api_client: Optional[Any] = None,
        nlu: Optional[NLU] = None,
        nlg: Optional[NLG] = None,
        seed_on_start: bool = True,
    ) -> None:
        self.db = database or Database()
        if seed_on_start:
            self.db.popular_dados_teste()

        self.api = api_client or criar_api_client()
        self.nlu = nlu or NLU()
        self.nlg = nlg or NLG()
        self.logic = BusinessLogic(
            database=self.db,
            api_client=self.api,
            nlg=self.nlg,
        )

    def iniciar_sessao(
        self,
        session_token: Optional[str] = None,
        usuario_identificador: Optional[str] = None,
    ) -> tuple[str, str]:
        """Cria ou recupera uma sessão e retorna (session_token, mensagem_boas_vindas)."""
        sessao = self.db.obter_ou_criar_sessao(
            session_token=session_token,
            usuario_identificador=usuario_identificador,
        )
        boas_vindas = self.nlg.boas_vindas()
        self.db.registrar_log(
            sessao_id=sessao.id,
            remetente="bot",
            mensagem=boas_vindas,
            intencao_detectada="boas_vindas",
            confianca_nlu=1.0,
            sentimento="POSITIVO",
            estado_apos_turno=sessao.estado_atual,
        )
        return sessao.session_token, boas_vindas

    def responder(self, mensagem: str, session_token: str) -> BotTurnResponse:
        """Processa um turno completo de conversa com persistência de estado e auditoria LGPD."""
        sessao = self.db.obter_ou_criar_sessao(session_token=session_token)
        estado_anterior = sessao.estado_atual

        # 1) Análise NLU (Eixo 1)
        nlu_res = self.nlu.analisar(mensagem)

        # 2) Regras de Negócio + Integração API + Máquina de Estados (Eixos 2, 3 e 4)
        outcome = self.logic.processar_turno(sessao, nlu_res)

        # 3) Atualiza SessaoUsuario (Stateful)
        self.db.atualizar_sessao(
            session_token=sessao.session_token,
            estado_atual=outcome.novo_estado,
            contexto=outcome.contexto_atualizado,
            falhas_consecutivas=outcome.falhas_consecutivas,
            transbordo_ativo=outcome.transbordo_ativo,
        )

        # 4) Auditoria ConversaLog com sanitização automática LGPD
        # (se o usuário acabou de pedir anonimização LGPD, não regravamos a entrada crua)
        if nlu_res.intencao != "anonimizar_lgpd":
            self.db.registrar_log(
                sessao_id=sessao.id,
                remetente="usuario",
                mensagem=mensagem,
                intencao_detectada=nlu_res.intencao,
                confianca_nlu=nlu_res.confianca,
                entidades=nlu_res.entidades,
                sentimento=nlu_res.sentimento,
                estado_apos_turno=outcome.novo_estado,
            )
            self.db.registrar_log(
                sessao_id=sessao.id,
                remetente="bot",
                mensagem=outcome.resposta,
                intencao_detectada=nlu_res.intencao,
                confianca_nlu=nlu_res.confianca,
                entidades={},
                sentimento=nlu_res.sentimento,
                estado_apos_turno=outcome.novo_estado,
            )

        debug_info = {
            "backend_nlp": nlu_res.backend_nlp,
            "texto_normalizado": nlu_res.texto_normalizado,
            "tokens": nlu_res.tokens,
            "tokens_sem_stopwords": nlu_res.tokens_sem_stopwords,
            "lemas": nlu_res.lemas,
            "entidades": sanitizar_entidades_lgpd(nlu_res.entidades),
            "ranking_intencoes": nlu_res.ranking_intencoes,
            "detalhes_extra": outcome.detalhes_extra,
        }

        return BotTurnResponse(
            session_token=sessao.session_token,
            resposta=outcome.resposta,
            estado_anterior=estado_anterior,
            estado_atual=outcome.novo_estado,
            intencao=nlu_res.intencao,
            confianca=nlu_res.confianca,
            sentimento=nlu_res.sentimento,
            score_insatisfacao=nlu_res.score_insatisfacao,
            falhas_consecutivas=outcome.falhas_consecutivas,
            transbordo_ativo=outcome.transbordo_ativo,
            origem_dados=outcome.origem_dados,
            debug_info=debug_info,
        )
