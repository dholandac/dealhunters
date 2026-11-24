"""
Eixo 4 — Regras de Negócio, Máquina de Estados (Stateful), Fallback e Transbordo Humano.

Orquestra as transições de `SessaoUsuario` com base na saída do `NLU`,
consulta o `CheapSharkClient` e o `Database` e delega a formatação ao `NLG`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import config
from src.api_client import CheapSharkClient
from src.database import Database, SessaoUsuario, mascarar_email
from src.nlg import NLG
from src.nlu import NLUResult, remover_acentos


@dataclass
class LogicOutcome:
    """Resultado de negócio de um turno antes da persistência final de logs."""

    resposta: str
    novo_estado: str
    contexto_atualizado: Dict[str, Any]
    falhas_consecutivas: int
    transbordo_ativo: bool = False
    origem_dados: str = "interno"  # interno | api_live | cache_fallback | database
    detalhes_extra: Dict[str, Any] = field(default_factory=dict)


class BusinessLogic:
    """
    Camada de Regras de Negócio e Máquina de Estados do Deal Hunters.

    Estados suportados:
      - INICIO
      - AGUARDANDO_JOGO
      - EXIBINDO_OFERTAS
      - AGUARDANDO_VALOR_ALERTA
      - AGUARDANDO_DADOS_PEDIDO
      - AGUARDANDO_CPF_VALIDACAO
      - TRANSBORDO_HUMANO
    """

    def __init__(
        self,
        database: Database,
        api_client: CheapSharkClient,
        nlg: NLG,
        max_fallbacks: int = config.MAX_CONSECUTIVE_FALLBACKS,
    ) -> None:
        self.db = database
        self.api = api_client
        self.nlg = nlg
        self.max_fallbacks = max_fallbacks

    # ------------------------------------------------------------------
    # Entrada principal de decisão por turno
    # ------------------------------------------------------------------

    def processar_turno(
        self, sessao: SessaoUsuario, nlu_res: NLUResult
    ) -> LogicOutcome:
        estado = sessao.estado_atual or "INICIO"
        contexto = sessao.get_contexto()
        texto_limpo = remover_acentos(nlu_res.texto_original.strip().lower())

        # 1) Comando de reinício após transbordo ou a qualquer momento
        if texto_limpo in {"reiniciar", "nova conversa", "comecar de novo", "reset"}:
            return LogicOutcome(
                resposta=self.nlg.boas_vindas(),
                novo_estado="INICIO",
                contexto_atualizado={},
                falhas_consecutivas=0,
                transbordo_ativo=False,
                origem_dados="interno",
            )

        # 2) Sessão já escalada para atendente humano
        if sessao.transbordo_ativo or estado == "TRANSBORDO_HUMANO":
            protocolo = contexto.get("protocolo_transbordo", "EM_ANDAMENTO")
            return LogicOutcome(
                resposta=self.nlg.sessao_em_transbordo(protocolo),
                novo_estado="TRANSBORDO_HUMANO",
                contexto_atualizado=contexto,
                falhas_consecutivas=sessao.falhas_consecutivas,
                transbordo_ativo=True,
                origem_dados="database",
            )

        # 3) Eixo 4 — Transbordo imediato por sentimento de extrema insatisfação ou urgência
        if nlu_res.sentimento == "CRITICO_URGENTE":
            return self._executar_transbordo(
                sessao=sessao,
                contexto=contexto,
                motivo="INSATISFACAO_URGENCIA",
                prioridade="ALTA",
                resumo=(
                    f"Sentimento crítico ({nlu_res.score_insatisfacao:.2f}) detectado "
                    f"na mensagem: '{nlu_res.texto_original}'"
                ),
            )

        # 4) Solicitação explícita de atendimento humano
        if nlu_res.intencao == "falar_atendente":
            return self._executar_transbordo(
                sessao=sessao,
                contexto=contexto,
                motivo="SOLICITACAO_CLIENTE",
                prioridade="NORMAL",
                resumo=f"Cliente solicitou atendimento humano: '{nlu_res.texto_original}'",
            )

        # 5) Intenções globais (não ficam presas em subestados)
        if nlu_res.intencao == "despedida":
            return LogicOutcome(
                resposta=self.nlg.despedida(),
                novo_estado="INICIO",
                contexto_atualizado={},
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        if nlu_res.intencao == "ajuda":
            return LogicOutcome(
                resposta=self.nlg.ajuda(),
                novo_estado=estado,
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        if nlu_res.intencao == "privacidade_lgpd":
            return LogicOutcome(
                resposta=self.nlg.politica_lgpd(),
                novo_estado=estado,
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        if nlu_res.intencao == "anonimizar_lgpd":
            qtd = self.db.anonimizar_sessao_lgpd(sessao.session_token)
            return LogicOutcome(
                resposta=self.nlg.confirmacao_anonimizacao_lgpd(qtd),
                novo_estado="INICIO",
                contexto_atualizado={},
                falhas_consecutivas=0,
                origem_dados="database",
                detalhes_extra={"registros_anonimizados": qtd},
            )

        # 6) Tratamento sensível ao estado atual (Máquina de Estados Stateful)
        if estado == "AGUARDANDO_JOGO":
            resultado_estado = self._tratar_aguardando_jogo(sessao, contexto, nlu_res)
            if resultado_estado is not None:
                return resultado_estado

        elif estado == "AGUARDANDO_VALOR_ALERTA":
            resultado_estado = self._tratar_aguardando_valor_alerta(
                sessao, contexto, nlu_res
            )
            if resultado_estado is not None:
                return resultado_estado

        elif estado == "AGUARDANDO_DADOS_PEDIDO":
            resultado_estado = self._tratar_aguardando_dados_pedido(
                contexto, nlu_res
            )
            if resultado_estado is not None:
                return resultado_estado

        elif estado == "AGUARDANDO_CPF_VALIDACAO":
            resultado_estado = self._tratar_aguardando_cpf(contexto, nlu_res)
            if resultado_estado is not None:
                return resultado_estado

        elif estado == "EXIBINDO_OFERTAS":
            resultado_estado = self._tratar_exibindo_ofertas(
                sessao, contexto, nlu_res
            )
            if resultado_estado is not None:
                return resultado_estado

        # 7) Despacho por intenção principal
        intencao = nlu_res.intencao

        if intencao == "saudacao":
            return LogicOutcome(
                resposta=self.nlg.saudacao_retorno(),
                novo_estado="INICIO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        if intencao == "consultar_preco":
            return self._fluxo_consultar_preco(contexto, nlu_res)

        if intencao == "buscar_promocoes":
            return self._fluxo_buscar_promocoes(contexto, nlu_res)

        if intencao == "selecionar_opcao":
            return self._fluxo_selecionar_opcao(contexto, nlu_res)

        if intencao == "criar_alerta":
            return self._fluxo_criar_alerta(sessao, contexto, nlu_res)

        if intencao == "consultar_pedido":
            return self._fluxo_consultar_pedido(contexto, nlu_res)

        if intencao == "solicitar_reembolso":
            return self._fluxo_reembolso(contexto, nlu_res)

        # 8) Eixo 4 — Tratamento de Fallback (2 falhas consecutivas -> Transbordo Humano)
        novas_falhas = sessao.falhas_consecutivas + 1
        if novas_falhas >= self.max_fallbacks:
            return self._executar_transbordo(
                sessao=sessao,
                contexto=contexto,
                motivo="FALLBACK_CONSECUTIVO",
                prioridade="NORMAL",
                resumo=(
                    f"NLU falhou por {novas_falhas} vezes consecutivas. "
                    f"Última mensagem: '{nlu_res.texto_original}'"
                ),
            )

        return LogicOutcome(
            resposta=self.nlg.primeiro_fallback(),
            novo_estado=estado,
            contexto_atualizado=contexto,
            falhas_consecutivas=novas_falhas,
            origem_dados="interno",
        )

    # ------------------------------------------------------------------
    # Handlers dos Estados da Máquina de Estados (Stateful)
    # ------------------------------------------------------------------

    def _tratar_aguardando_jogo(
        self,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        nlu_res: NLUResult,
    ) -> Optional[LogicOutcome]:
        # Permite trocar de assunto se o usuário disparar claramente outra intenção
        if nlu_res.intencao in {
            "consultar_pedido",
            "buscar_promocoes",
            "solicitar_reembolso",
        }:
            return None

        jogo = nlu_res.entidades.get("jogo")
        if not jogo:
            # Aceita resposta direta do usuário como título do jogo (ex.: "Portal 2")
            texto_bruto = nlu_res.texto_original.strip(" ?!.,")
            if 2 <= len(texto_bruto) <= 50 and len(texto_bruto.split()) <= 7:
                jogo = texto_bruto

        if not jogo:
            return None

        acao = contexto.get("acao_pos_jogo", "preco")
        if acao == "alerta":
            valor = nlu_res.entidades.get("valor_brl") or contexto.get(
                "valor_alerta_pendente"
            )
            if valor is not None:
                return self._registrar_alerta_final(
                    sessao, contexto, jogo, float(valor), nlu_res.entidades.get("email")
                )
            novo_ctx = dict(contexto)
            novo_ctx["jogo_alerta"] = jogo
            novo_ctx["ultimo_jogo"] = jogo
            return LogicOutcome(
                resposta=self.nlg.pedir_valor_alerta(jogo),
                novo_estado="AGUARDANDO_VALOR_ALERTA",
                contexto_atualizado=novo_ctx,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        return self._executar_busca_jogo(
            contexto, jogo, loja_filtro=nlu_res.entidades.get("loja")
        )

    def _tratar_aguardando_valor_alerta(
        self,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        nlu_res: NLUResult,
    ) -> Optional[LogicOutcome]:
        valor = nlu_res.entidades.get("valor_brl")
        if valor is None:
            m_num = re.search(r"(\d+(?:[.,]\d{1,2})?)", nlu_res.texto_original)
            if m_num:
                try:
                    valor = float(m_num.group(1).replace(",", "."))
                except ValueError:
                    valor = None

        if valor is None:
            # Se o usuário mudou de assunto, deixa o despacho normal tratar
            if nlu_res.intencao != "desconhecida":
                return None
            return None

        jogo = contexto.get("jogo_alerta") or contexto.get("ultimo_jogo")
        if not jogo:
            return LogicOutcome(
                resposta=self.nlg.pedir_nome_jogo(motivo="alerta"),
                novo_estado="AGUARDANDO_JOGO",
                contexto_atualizado={
                    **contexto,
                    "acao_pos_jogo": "alerta",
                    "valor_alerta_pendente": valor,
                },
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        return self._registrar_alerta_final(
            sessao, contexto, jogo, float(valor), nlu_res.entidades.get("email")
        )

    def _tratar_aguardando_dados_pedido(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> Optional[LogicOutcome]:
        codigo = nlu_res.entidades.get("codigo_pedido")
        if not codigo:
            m_num = re.fullmatch(r"#?(\d{3,6})", nlu_res.texto_original.strip())
            if m_num:
                codigo = f"DH-{m_num.group(1)}"

        if not codigo:
            if nlu_res.intencao not in {"desconhecida", "consultar_pedido"}:
                return None
            return None

        return self._consultar_pedido_com_codigo(
            contexto, codigo, cpf=nlu_res.entidades.get("cpf")
        )

    def _tratar_aguardando_cpf(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> Optional[LogicOutcome]:
        cpf = nlu_res.entidades.get("cpf")
        codigo = contexto.get("pedido_em_consulta")

        if not cpf or not codigo:
            if nlu_res.intencao != "desconhecida":
                return None
            return None

        pedido = self.db.validar_cpf_pedido(codigo, cpf)
        if pedido is None:
            return LogicOutcome(
                resposta=self.nlg.cpf_invalido_pedido(codigo),
                novo_estado="AGUARDANDO_CPF_VALIDACAO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="database",
            )

        novo_ctx = dict(contexto)
        novo_ctx.pop("pedido_em_consulta", None)
        novo_ctx["ultimo_pedido_validado"] = pedido.codigo_pedido
        return LogicOutcome(
            resposta=self.nlg.detalhes_pedido(pedido),
            novo_estado="INICIO",
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados="database",
        )

    def _tratar_exibindo_ofertas(
        self,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        nlu_res: NLUResult,
    ) -> Optional[LogicOutcome]:
        # Permite selecionar opção apenas digitando o número (ex.: "1" ou "2")
        texto_strip = nlu_res.texto_original.strip()
        if texto_strip.isdigit() and 1 <= int(texto_strip) <= 5:
            nlu_res.entidades["opcao_numero"] = int(texto_strip)
            return self._fluxo_selecionar_opcao(contexto, nlu_res)

        if nlu_res.intencao == "selecionar_opcao":
            return self._fluxo_selecionar_opcao(contexto, nlu_res)

        # Filtro contextual sobre o último jogo exibido (ex.: "só na Steam")
        loja = nlu_res.entidades.get("loja")
        valor = nlu_res.entidades.get("valor_brl")
        jogo_novo = nlu_res.entidades.get("jogo")
        ultimo_jogo = contexto.get("ultimo_jogo")

        if (
            loja
            and not jogo_novo
            and valor is None
            and ultimo_jogo
            and nlu_res.intencao in {"buscar_promocoes", "desconhecida", "consultar_preco"}
        ):
            return self._executar_busca_jogo(contexto, ultimo_jogo, loja_filtro=loja)

        if nlu_res.intencao == "criar_alerta":
            return self._fluxo_criar_alerta(sessao, contexto, nlu_res)

        return None

    # ------------------------------------------------------------------
    # Fluxos Principais de Negócio
    # ------------------------------------------------------------------

    def _fluxo_consultar_preco(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> LogicOutcome:
        jogo = nlu_res.entidades.get("jogo")
        loja = nlu_res.entidades.get("loja")

        if not jogo:
            novo_ctx = dict(contexto)
            novo_ctx["acao_pos_jogo"] = "preco"
            return LogicOutcome(
                resposta=self.nlg.pedir_nome_jogo(motivo="preco"),
                novo_estado="AGUARDANDO_JOGO",
                contexto_atualizado=novo_ctx,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        return self._executar_busca_jogo(contexto, jogo, loja_filtro=loja)

    def _executar_busca_jogo(
        self,
        contexto: Dict[str, Any],
        jogo: str,
        loja_filtro: Optional[str] = None,
    ) -> LogicOutcome:
        resultado = self.api.consultar_preco_jogo(
            titulo=jogo, loja_filtro=loja_filtro, limite_ofertas=4
        )
        dados = resultado.dados or {"jogo": jogo, "ofertas": []}
        ofertas = dados.get("ofertas", [])

        novo_ctx = dict(contexto)
        novo_ctx.pop("acao_pos_jogo", None)
        novo_ctx["ultimo_jogo"] = dados.get("jogo", jogo)
        novo_ctx["ultimas_ofertas"] = ofertas

        resposta = self.nlg.ofertas_jogo(
            dados=dados,
            origem=resultado.origem,
            cotacao=resultado.cotacao_usd_brl,
            origem_cotacao=resultado.origem_cotacao,
            aviso_fallback=resultado.erro_amigavel,
        )
        novo_estado = "EXIBINDO_OFERTAS" if ofertas else "INICIO"

        return LogicOutcome(
            resposta=resposta,
            novo_estado=novo_estado,
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados=resultado.origem,
            detalhes_extra={
                "cotacao_usd_brl": resultado.cotacao_usd_brl,
                "origem_cotacao": resultado.origem_cotacao,
                "qtd_ofertas": len(ofertas),
            },
        )

    def _fluxo_buscar_promocoes(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> LogicOutcome:
        preco_max = nlu_res.entidades.get("valor_brl")
        loja = nlu_res.entidades.get("loja")
        jogo = nlu_res.entidades.get("jogo")

        resultado = self.api.buscar_promocoes(
            preco_max_brl=preco_max,
            loja_filtro=loja,
            titulo_filtro=jogo,
            limite=5,
        )
        dados = resultado.dados or {"ofertas": []}
        ofertas = dados.get("ofertas", [])

        novo_ctx = dict(contexto)
        novo_ctx["ultimas_ofertas"] = ofertas
        if jogo:
            novo_ctx["ultimo_jogo"] = jogo
        elif ofertas:
            novo_ctx["ultimo_jogo"] = ofertas[0]["titulo"]

        resposta = self.nlg.promocoes(
            dados=dados,
            origem=resultado.origem,
            cotacao=resultado.cotacao_usd_brl,
            origem_cotacao=resultado.origem_cotacao,
            aviso_fallback=resultado.erro_amigavel,
        )
        novo_estado = "EXIBINDO_OFERTAS" if ofertas else "INICIO"

        return LogicOutcome(
            resposta=resposta,
            novo_estado=novo_estado,
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados=resultado.origem,
            detalhes_extra={
                "cotacao_usd_brl": resultado.cotacao_usd_brl,
                "qtd_ofertas": len(ofertas),
            },
        )

    def _fluxo_selecionar_opcao(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> LogicOutcome:
        ofertas = contexto.get("ultimas_ofertas", [])
        indice = nlu_res.entidades.get("opcao_numero")

        if not ofertas or indice is None or not (1 <= int(indice) <= len(ofertas)):
            return LogicOutcome(
                resposta=self.nlg.opcao_invalida(len(ofertas)),
                novo_estado="EXIBINDO_OFERTAS" if ofertas else "INICIO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        idx = int(indice)
        oferta = ofertas[idx - 1]
        novo_ctx = dict(contexto)
        novo_ctx["ultimo_jogo"] = oferta.get("titulo", contexto.get("ultimo_jogo"))

        return LogicOutcome(
            resposta=self.nlg.detalhe_oferta(oferta, idx),
            novo_estado="EXIBINDO_OFERTAS",
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados="interno",
        )

    def _fluxo_criar_alerta(
        self,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        nlu_res: NLUResult,
    ) -> LogicOutcome:
        texto_norm = remover_acentos(nlu_res.texto_normalizado.lower())
        if any(p in texto_norm for p in ("listar", "meus alertas", "ver alertas")):
            alertas = self.db.listar_alertas_sessao(sessao.id)
            return LogicOutcome(
                resposta=self.nlg.lista_alertas(alertas),
                novo_estado=sessao.estado_atual or "INICIO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="database",
            )

        jogo = nlu_res.entidades.get("jogo") or contexto.get("ultimo_jogo")
        valor = nlu_res.entidades.get("valor_brl")
        email = nlu_res.entidades.get("email")

        if not jogo:
            novo_ctx = dict(contexto)
            novo_ctx["acao_pos_jogo"] = "alerta"
            if valor is not None:
                novo_ctx["valor_alerta_pendente"] = valor
            return LogicOutcome(
                resposta=self.nlg.pedir_nome_jogo(motivo="alerta"),
                novo_estado="AGUARDANDO_JOGO",
                contexto_atualizado=novo_ctx,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        if valor is None:
            novo_ctx = dict(contexto)
            novo_ctx["jogo_alerta"] = jogo
            novo_ctx["ultimo_jogo"] = jogo
            return LogicOutcome(
                resposta=self.nlg.pedir_valor_alerta(jogo),
                novo_estado="AGUARDANDO_VALOR_ALERTA",
                contexto_atualizado=novo_ctx,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        return self._registrar_alerta_final(sessao, contexto, jogo, float(valor), email)

    def _registrar_alerta_final(
        self,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        jogo: str,
        valor_brl: float,
        email: Optional[str],
    ) -> LogicOutcome:
        self.db.criar_alerta_preco(
            sessao_id=sessao.id,
            jogo_titulo=jogo,
            preco_alvo_brl=valor_brl,
            contato=email,
        )
        novo_ctx = dict(contexto)
        novo_ctx.pop("jogo_alerta", None)
        novo_ctx.pop("acao_pos_jogo", None)
        novo_ctx.pop("valor_alerta_pendente", None)
        novo_ctx["ultimo_jogo"] = jogo

        contato_seguro = mascarar_email(email) if email else None
        return LogicOutcome(
            resposta=self.nlg.alerta_criado(jogo, valor_brl, contato_seguro),
            novo_estado="INICIO",
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados="database",
        )

    def _fluxo_consultar_pedido(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> LogicOutcome:
        codigo = nlu_res.entidades.get("codigo_pedido")
        cpf = nlu_res.entidades.get("cpf")

        if not codigo:
            return LogicOutcome(
                resposta=self.nlg.pedir_codigo_pedido(),
                novo_estado="AGUARDANDO_DADOS_PEDIDO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="interno",
            )

        return self._consultar_pedido_com_codigo(contexto, codigo, cpf=cpf)

    def _consultar_pedido_com_codigo(
        self,
        contexto: Dict[str, Any],
        codigo_pedido: str,
        cpf: Optional[str] = None,
    ) -> LogicOutcome:
        pedido = self.db.buscar_pedido_por_codigo(codigo_pedido)
        if pedido is None:
            return LogicOutcome(
                resposta=self.nlg.pedido_nao_encontrado(codigo_pedido),
                novo_estado="AGUARDANDO_DADOS_PEDIDO",
                contexto_atualizado=contexto,
                falhas_consecutivas=0,
                origem_dados="database",
            )

        if cpf:
            validado = self.db.validar_cpf_pedido(codigo_pedido, cpf)
            if validado is None:
                novo_ctx = {**contexto, "pedido_em_consulta": pedido.codigo_pedido}
                return LogicOutcome(
                    resposta=self.nlg.cpf_invalido_pedido(pedido.codigo_pedido),
                    novo_estado="AGUARDANDO_CPF_VALIDACAO",
                    contexto_atualizado=novo_ctx,
                    falhas_consecutivas=0,
                    origem_dados="database",
                )
            return LogicOutcome(
                resposta=self.nlg.detalhes_pedido(validado),
                novo_estado="INICIO",
                contexto_atualizado={
                    **contexto,
                    "ultimo_pedido_validado": validado.codigo_pedido,
                },
                falhas_consecutivas=0,
                origem_dados="database",
            )

        novo_ctx = {**contexto, "pedido_em_consulta": pedido.codigo_pedido}
        return LogicOutcome(
            resposta=self.nlg.pedir_cpf_validacao(pedido.codigo_pedido),
            novo_estado="AGUARDANDO_CPF_VALIDACAO",
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=0,
            origem_dados="database",
        )

    def _fluxo_reembolso(
        self, contexto: Dict[str, Any], nlu_res: NLUResult
    ) -> LogicOutcome:
        codigo = nlu_res.entidades.get("codigo_pedido") or contexto.get(
            "ultimo_pedido_validado"
        )
        pedido = self.db.buscar_pedido_por_codigo(codigo) if codigo else None
        return LogicOutcome(
            resposta=self.nlg.orientacao_reembolso(pedido),
            novo_estado="INICIO",
            contexto_atualizado=contexto,
            falhas_consecutivas=0,
            origem_dados="database" if pedido else "interno",
        )

    def _executar_transbordo(
        self,
        *,
        sessao: SessaoUsuario,
        contexto: Dict[str, Any],
        motivo: str,
        prioridade: str,
        resumo: str,
    ) -> LogicOutcome:
        evento = self.db.registrar_transbordo(
            sessao_id=sessao.id,
            motivo=motivo,
            prioridade=prioridade,
            resumo_contexto=resumo,
        )
        novo_ctx = dict(contexto)
        novo_ctx["protocolo_transbordo"] = evento.protocolo
        novo_ctx["motivo_transbordo"] = motivo

        resposta = self.nlg.transbordo_humano(
            protocolo=evento.protocolo,
            motivo=motivo,
            prioridade=prioridade,
            ultimo_jogo=contexto.get("ultimo_jogo"),
            codigo_pedido=contexto.get("pedido_em_consulta")
            or contexto.get("ultimo_pedido_validado"),
        )
        return LogicOutcome(
            resposta=resposta,
            novo_estado="TRANSBORDO_HUMANO",
            contexto_atualizado=novo_ctx,
            falhas_consecutivas=sessao.falhas_consecutivas,
            transbordo_ativo=True,
            origem_dados="database",
            detalhes_extra={
                "protocolo": evento.protocolo,
                "motivo": motivo,
                "prioridade": prioridade,
            },
        )
