"""Formatação das respostas do assistente Hunter."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import config


def fmt_brl(valor: float) -> str:
    """Formata valor monetário no padrão brasileiro (R$ 49,90)."""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_usd(valor: float) -> str:
    """Formata valor em dólares (US$ 9.99)."""
    return f"US$ {valor:.2f}"


class NLG:
    """Gerador de linguagem natural desacoplado para a persona Hunter."""

    def __init__(
        self,
        app_name: str = config.APP_NAME,
        bot_name: str = config.BOT_NAME,
    ) -> None:
        self.app_name = app_name
        self.bot_name = bot_name

    def boas_vindas(self) -> str:
        return (
            f"Olá! Eu sou o {self.bot_name}, assistente virtual oficial da {self.app_name}.\n"
            "Posso ajudar você com:\n"
            "  1. Consultar e comparar preços de jogos em tempo real (Steam, GOG, Epic, etc.)\n"
            "  2. Buscar promoções filtradas pelo seu orçamento ou loja favorita\n"
            "  3. Criar alertas de preço para ser avisado quando um jogo baixar\n"
            "  4. Consultar pedidos, chaves de ativação, reembolso e privacidade (LGPD)\n\n"
            "Como posso ajudar você hoje?"
        )

    def ajuda(self) -> str:
        return (
            f"Aqui estão alguns exemplos do que você pode me perguntar na {self.app_name}:\n"
            '  - "Quanto custa Hollow Knight?"\n'
            '  - "Tem promoção até 50 reais na Steam?"\n'
            '  - "Me avisa quando Elden Ring baixar para R$ 150"\n'
            '  - "Quero consultar meu pedido DH-1001"\n'
            '  - "Como vocês protegem meus dados (LGPD)?"\n'
            '  - "Quero falar com um atendente humano"'
        )

    def despedida(self) -> str:
        return (
            f"Obrigado por contar com a {self.app_name}! "
            "Sempre que quiser caçar uma nova oferta ou acompanhar um pedido, "
            "estarei por aqui. Bons jogos!"
        )

    def saudacao_retorno(self) -> str:
        return (
            f"Olá! Tudo bem? Sou o {self.bot_name}, da {self.app_name}. "
            "Qual jogo, promoção ou pedido você gostaria de consultar agora?"
        )

    def pedir_nome_jogo(self, motivo: str = "preco") -> str:
        if motivo == "alerta":
            return (
                "Com prazer! Para qual jogo você deseja configurar um alerta de preço?"
            )
        return (
            "Com certeza! Qual é o nome do jogo cujo preço você gostaria de consultar?"
        )

    def pedir_valor_alerta(self, jogo: str) -> str:
        return (
            f"Ótima escolha! Qual é o preço máximo em reais (ex.: R$ 50) "
            f"que você deseja pagar por {jogo}?"
        )

    def ofertas_jogo(
        self,
        dados: Dict[str, Any],
        origem: str,
        cotacao: float,
        origem_cotacao: str,
        aviso_fallback: Optional[str] = None,
    ) -> str:
        jogo = dados.get("jogo", "o jogo consultado")
        ofertas: List[Dict[str, Any]] = dados.get("ofertas", [])

        linhas: List[str] = []
        if aviso_fallback:
            linhas.append(f"[!] {aviso_fallback}\n")

        if not ofertas:
            return self.jogo_nao_encontrado(jogo, aviso_fallback)

        linhas.append(f"Encontrei as melhores ofertas para {jogo}:")
        for idx, oferta in enumerate(ofertas, start=1):
            desconto = oferta.get("desconto_pct", 0.0)
            tag_desc = f" (-{desconto:.0f}%)" if desconto >= 1.0 else ""
            preco_usd = oferta.get("preco_usd")
            usd_info = f" ({fmt_usd(preco_usd)})" if preco_usd is not None else ""
            linhas.append(
                f"  [{idx}] {oferta['loja']}: {fmt_brl(oferta['preco_brl'])}"
                f"{usd_info}{tag_desc}"
            )

        menor_brl = dados.get("menor_preco_historico_brl")
        menor_usd = dados.get("menor_preco_historico_usd")
        if menor_brl is not None:
            menor_usd_info = f" ({fmt_usd(menor_usd)})" if menor_usd is not None else ""
            linhas.append(
                f"\nMenor preço histórico registrado: {fmt_brl(menor_brl)}{menor_usd_info}."
            )

        if origem == "api_live" and origem_cotacao == "brl_nativo":
            linhas.append(
                "Fonte: IsThereAnyDeal API (preço real em BRL — Brasil)."
            )
        else:
            fonte_label = (
                "CheapShark API (tempo real)"
                if origem == "api_live"
                else "Catálogo local de contingência"
            )
            cotacao_label = (
                "tempo real" if origem_cotacao == "awesomeapi_live" else "referência"
            )
            taxa_val = cotacao if cotacao is not None else config.DEFAULT_USD_BRL_RATE
            linhas.append(
                f"Fonte: {fonte_label} | Câmbio USD->BRL ({cotacao_label}): R$ {taxa_val:.2f}."
            )
        linhas.append(
            "\nPróximo passo: digite o número da opção (ex.: 'opção 1') para ver o link "
            "da oferta, filtre por loja (ex.: 'só na Steam') ou peça um alerta de preço."
        )
        return "\n".join(linhas)

    def jogo_nao_encontrado(
        self, titulo: str, aviso_fallback: Optional[str] = None
    ) -> str:
        prefixo = f"[!] {aviso_fallback}\n\n" if aviso_fallback else ""
        return (
            f"{prefixo}Não encontrei ofertas ativas para '{titulo}' no momento. "
            "Você pode tentar o título em inglês (como aparece na Steam) ou pedir "
            "para listar as promoções do dia."
        )

    def promocoes(
        self,
        dados: Dict[str, Any],
        origem: str,
        cotacao: float,
        origem_cotacao: str,
        aviso_fallback: Optional[str] = None,
    ) -> str:
        ofertas: List[Dict[str, Any]] = dados.get("ofertas", [])
        preco_max = dados.get("filtro_preco_max_brl")
        loja = dados.get("filtro_loja")

        linhas: List[str] = []
        if aviso_fallback:
            linhas.append(f"[!] {aviso_fallback}\n")

        filtros_desc: List[str] = []
        if preco_max is not None:
            filtros_desc.append(f"até {fmt_brl(preco_max)}")
        if loja:
            filtros_desc.append(f"na loja {loja}")
        sufixo = f" ({', '.join(filtros_desc)})" if filtros_desc else ""

        if not ofertas:
            return (
                f"{''.join(linhas)}Não encontrei promoções ativas{sufixo} agora. "
                "Tente aumentar um pouco o orçamento ou remover o filtro de loja."
            )

        linhas.append(f"Destaques de promoções disponíveis{sufixo}:")
        for idx, oferta in enumerate(ofertas, start=1):
            desconto = oferta.get("desconto_pct", 0.0)
            meta = f" | Metacritic {oferta['metacritic']}" if oferta.get("metacritic") else ""
            preco_usd = oferta.get("preco_usd")
            if preco_usd is not None:
                detalhe_preco = f" ({fmt_usd(preco_usd)}, -{desconto:.0f}%)"
            else:
                detalhe_preco = f" (-{desconto:.0f}%)" if desconto >= 1.0 else ""
            linhas.append(
                f"  [{idx}] {oferta['titulo']} — {fmt_brl(oferta['preco_brl'])}"
                f"{detalhe_preco} na {oferta['loja']}{meta}"
            )

        if origem == "api_live" and origem_cotacao == "brl_nativo":
            linhas.append(
                "\nFonte: IsThereAnyDeal API (preço real em BRL — Brasil)."
            )
        else:
            fonte_label = (
                "CheapShark API (tempo real)"
                if origem == "api_live"
                else "Catálogo local de contingência"
            )
            cotacao_label = (
                "tempo real" if origem_cotacao == "awesomeapi_live" else "referência"
            )
            taxa_val = cotacao if cotacao is not None else config.DEFAULT_USD_BRL_RATE
            linhas.append(
                f"\nFonte: {fonte_label} | Câmbio USD->BRL ({cotacao_label}): R$ {taxa_val:.2f}."
            )
        linhas.append(
            "Digite o número de uma opção (ex.: 'opção 2') para ver detalhes e link oficial."
        )
        return "\n".join(linhas)

    def detalhe_oferta(self, oferta: Dict[str, Any], indice: int) -> str:
        preco_usd = oferta.get("preco_usd")
        preco_normal_usd = oferta.get("preco_normal_usd")
        usd_promo = f" ({fmt_usd(preco_usd)})" if preco_usd is not None else ""
        usd_normal = (
            f" ({fmt_usd(preco_normal_usd)})" if preco_normal_usd is not None else ""
        )
        linhas = [
            f"Detalhes da opção [{indice}] — {oferta['titulo']}:",
            f"  - Loja parceira: {oferta['loja']}",
            f"  - Preço promocional: {fmt_brl(oferta['preco_brl'])}{usd_promo}",
            f"  - Preço original: {fmt_brl(oferta['preco_normal_brl'])}{usd_normal}",
            f"  - Economia: {oferta.get('desconto_pct', 0.0):.1f}%",
        ]
        if oferta.get("metacritic"):
            linhas.append(f"  - Nota Metacritic: {oferta['metacritic']}/100")
        linhas.append(f"  - Link oficial da oferta: {oferta['link']}")
        linhas.append(
            "\nSe preferir aguardar um desconto maior, basta dizer: "
            "'criar alerta de preço até R$ ...'."
        )
        return "\n".join(linhas)

    def opcao_invalida(self, maximo: int) -> str:
        if maximo <= 0:
            return (
                "Ainda não há uma lista de ofertas aberta nesta sessão. "
                "Pesquise um jogo ou peça as promoções do dia primeiro!"
            )
        return (
            f"Essa opção não está na lista atual. Escolha um número entre 1 e {maximo}."
        )

    def alerta_criado(
        self,
        jogo: str,
        preco_alvo_brl: float,
        contato_mascarado: Optional[str] = None,
    ) -> str:
        msg = (
            f"Alerta registrado com sucesso! Vou monitorar {jogo} e avisar "
            f"assim que atingir {fmt_brl(preco_alvo_brl)} ou menos."
        )
        if contato_mascarado:
            msg += f"\nNotificação vinculada com proteção LGPD para: {contato_mascarado}."
        return msg

    def lista_alertas(self, alertas: List[Any]) -> str:
        if not alertas:
            return (
                "Você ainda não possui alertas de preço ativos nesta sessão. "
                "Diga, por exemplo: 'Me avisa quando Hollow Knight baixar para R$ 30'."
            )
        linhas = ["Seus alertas de preço ativos:"]
        for idx, alerta in enumerate(alertas, start=1):
            linhas.append(
                f"  [{idx}] {alerta.jogo_titulo} — alvo: {fmt_brl(alerta.preco_alvo_brl)}"
            )
        return "\n".join(linhas)

    def pedir_codigo_pedido(self) -> str:
        return (
            "Claro! Para consultar seu pedido digital, informe o código do pedido "
            "(ex.: DH-1001)."
        )

    def pedir_cpf_validacao(self, codigo_pedido: str) -> str:
        return (
            f"Encontrei o pedido {codigo_pedido}. Por segurança e conformidade com a LGPD, "
            "informe o CPF do titular da compra para validarmos sua identidade "
            "(seu CPF será validado via hash e mascarado nos registros de auditoria)."
        )

    def pedido_nao_encontrado(self, codigo_pedido: str) -> str:
        return (
            f"Não localizei nenhum pedido com o código {codigo_pedido}. "
            "Confira se o código está no formato DH-1001 ou peça para falar com um atendente."
        )

    def cpf_invalido_pedido(self, codigo_pedido: str) -> str:
        return (
            f"O CPF informado não confere com o titular do pedido {codigo_pedido}. "
            "Por segurança, tente novamente ou solicite atendimento humano."
        )

    def detalhes_pedido(self, pedido: Any) -> str:
        status_descricoes = {
            "ENTREGUE": "Entregue (chave liberada)",
            "CHAVE_PENDENTE": "Pagamento aprovado — chave em emissão pelo distribuidor",
            "AGUARDANDO_PAGAMENTO": "Aguardando confirmação do pagamento",
            "REEMBOLSO_EM_ANALISE": "Solicitação de reembolso em análise financeira",
        }
        status_texto = status_descricoes.get(pedido.status, pedido.status)
        linhas = [
            f"Identidade confirmada! Resumo do pedido {pedido.codigo_pedido}:",
            f"  - Titular: {pedido.cliente_nome} (CPF {pedido.cpf_mascarado})",
            f"  - Jogo: {pedido.jogo_titulo} ({pedido.plataforma})",
            f"  - Valor pago: {fmt_brl(pedido.valor_brl)}",
            f"  - Status: {status_texto}",
        ]
        if pedido.chave_ativacao:
            linhas.append(f"  - Chave de ativação: {pedido.chave_ativacao}")
        elif pedido.status == "CHAVE_PENDENTE":
            linhas.append(
                "  - Previsão: chaves pendentes costumam ser liberadas em até 30 minutos. "
                "Se precisar de ajuda prioritária, posso transferir você para um atendente."
            )
        elif pedido.status == "AGUARDANDO_PAGAMENTO":
            linhas.append(
                "  - Assim que o pagamento for compensado, a chave é liberada automaticamente."
            )
        elif pedido.status == "REEMBOLSO_EM_ANALISE":
            linhas.append(
                "  - O estorno está em andamento com prazo médio de até 5 dias úteis."
            )
        return "\n".join(linhas)

    def orientacao_reembolso(self, pedido: Optional[Any] = None) -> str:
        if pedido is not None:
            if pedido.status == "REEMBOLSO_EM_ANALISE":
                return (
                    f"O pedido {pedido.codigo_pedido} ({pedido.jogo_titulo}) já está com "
                    "reembolso em análise financeira. Se desejar agilizar ou falar com "
                    "nossa equipe, basta dizer 'falar com atendente'."
                )
            return (
                f"Localizei o pedido {pedido.codigo_pedido} ({pedido.jogo_titulo}). "
                "Como solicitações de reembolso exigem validação da chave não resgatada "
                "pelo time financeiro, posso transferir você agora mesmo para um especialista. "
                "Deseja que eu transfira para um atendente humano?"
            )
        return (
            "Entendo que você deseja solicitar um reembolso. "
            "Na Deal Hunters, pedidos com chave não ativada podem ser estornados em até 7 dias. "
            "Informe o código do pedido (ex.: DH-1001) ou diga 'falar com atendente' para "
            "iniciarmos o atendimento humano."
        )

    def politica_lgpd(self) -> str:
        return (
            f"Na {self.app_name}, levamos a LGPD (Lei nº 13.709/2018) a sério:\n"
            "  - CPFs são validados por hash criptográfico (SHA-256) e nunca salvos em texto claro.\n"
            "  - CPFs e e-mails enviados no chat são automaticamente mascarados na tabela ConversaLog.\n"
            "  - Você pode exercer o direito de anonimização a qualquer momento dizendo "
            "'apagar meus dados'."
        )

    def confirmacao_anonimizacao_lgpd(self, qtd_registros: int) -> str:
        return (
            "Solicitação LGPD concluída com sucesso! "
            f"{qtd_registros} registro(s) desta sessão foram anonimizados no banco de auditoria "
            "(ConversaLog), preservando apenas métricas não identificáveis."
        )

    def primeiro_fallback(self) -> str:
        return (
            "Desculpe, não consegui entender com clareza o que você precisa. "
            "Você poderia reformular sua mensagem?\n"
            "Exemplos: 'Quanto custa Hollow Knight?', 'Promoções até R$ 50 na Steam' "
            "ou 'Consultar pedido DH-1001'."
        )

    def transbordo_humano(
        self,
        protocolo: str,
        motivo: str,
        prioridade: str,
        ultimo_jogo: Optional[str] = None,
        codigo_pedido: Optional[str] = None,
    ) -> str:
        if motivo == "INSATISFACAO_URGENCIA":
            abertura = (
                "Sinto muito pelo transtorno que você está enfrentando. "
                "Identifiquei urgência no seu relato e já acionei nosso suporte humano prioritário!"
            )
        elif motivo == "FALLBACK_CONSECUTIVO":
            abertura = (
                "Percebi que não consegui compreender sua solicitação nas últimas tentativas. "
                "Para não fazer você perder tempo, estou transferindo seu atendimento "
                "para um especialista humano!"
            )
        else:
            abertura = (
                "Perfeito! Já estou transferindo sua conversa de forma transparente "
                "para um de nossos especialistas humanos."
            )

        resumo_itens: List[str] = []
        if ultimo_jogo:
            resumo_itens.append(f"Jogo em contexto: {ultimo_jogo}")
        if codigo_pedido:
            resumo_itens.append(f"Pedido em contexto: {codigo_pedido}")
        contexto_str = (
            f"\n  - Contexto repassado: {' | '.join(resumo_itens)}"
            if resumo_itens
            else ""
        )

        return (
            f"{abertura}\n\n"
            "=== TRANSBORDO PARA ATENDIMENTO HUMANO ===\n"
            f"  - Protocolo registrado: {protocolo}\n"
            f"  - Motivo: {motivo}\n"
            f"  - Prioridade na fila: {prioridade}"
            f"{contexto_str}\n"
            "  - Status: Evento gravado em banco de dados; um atendente assumirá em instantes.\n"
            "=========================================\n"
            "(Dica: digite 'reiniciar' se desejar iniciar um novo atendimento com o bot.)"
        )

    def sessao_em_transbordo(self, protocolo: str) -> str:
        return (
            f"Seu atendimento (protocolo {protocolo}) já foi encaminhado para a fila humana. "
            "Aguarde o contato do especialista ou digite 'reiniciar' para começar uma nova conversa comigo."
        )
