"""
Arquivo Mestre do Projeto Deal Hunters — Execução Interativa no Terminal (Live Demo).

Uso:
    python main.py                  # Modo padrão (API CheapShark em tempo real)
    python main.py --debug          # Exibe pipeline NLU, lemas, NER e estados a cada turno
    python main.py --offline        # Simula contingência offline (resiliência de rede)
    python main.py --session ID     # Retoma uma sessão persistida no SQLite (Stateful)
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid

import config
from src.api_client import CheapSharkClient
from src.chatbot import DealHuntersBot
from src.database import Database


def imprimir_banner(backend_nlp: str, offline: bool, session_token: str) -> None:
    modo_api = "CONTINGÊNCIA OFFLINE (cache local)" if offline else "LIVE (CheapShark + AwesomeAPI)"
    print("=" * 72)
    print(f"  {config.APP_NAME.upper()} — {config.COMPANY_SLOGAN}")
    print("=" * 72)
    print(f"  NLP Engine : {backend_nlp}")
    print(f"  API REST   : {modo_api}")
    print(f"  Banco ORM  : {config.DATABASE_URL}")
    print(f"  Sessão ID  : {session_token}")
    print("-" * 72)
    print("  Comandos de demonstração no terminal:")
    print("    /debug        -> liga/desliga telemetria NLU (lemas, NER, intenção, estado)")
    print("    /sessao       -> inspeciona registro atual na tabela SessaoUsuario")
    print("    /historico    -> exibe últimos logs LGPD na tabela ConversaLog")
    print("    /transbordos  -> lista protocolos registrados em TransbordoHumano")
    print("    /reset        -> inicia uma nova sessão limpa (útil entre cenários da demo)")
    print("    /sair         -> encerra o atendimento")
    print("=" * 72)
    print()


def imprimir_debug(resp) -> None:
    dbg = resp.debug_info
    print("\n  [DEBUG TÉCNICO]")
    print(f"    NLP Backend  : {dbg.get('backend_nlp')}")
    print(f"    Normalizado  : {dbg.get('texto_normalizado')}")
    print(f"    Lemas (spaCy): {dbg.get('lemas')}")
    print(f"    Intenção     : {resp.intencao} (confiança: {resp.confianca:.2f})")
    print(f"    Entidades NER: {json.dumps(dbg.get('entidades', {}), ensure_ascii=False)}")
    print(f"    Sentimento   : {resp.sentimento} (insatisfação: {resp.score_insatisfacao:.2f})")
    print(f"    Estado       : {resp.estado_anterior} -> {resp.estado_atual}")
    print(f"    Fallbacks    : {resp.falhas_consecutivas}/{config.MAX_CONSECUTIVE_FALLBACKS}")
    print(f"    Origem dados : {resp.origem_dados}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chatbot Corporativo Deal Hunters (Terminal CLI)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe dados internos do NLU, NER e Máquina de Estados em cada turno.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Força modo offline para demonstrar resiliência com catálogo de contingência.",
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Token de sessão existente no SQLite para retomar conversa Stateful.",
    )
    args = parser.parse_args()

    offline = args.offline or config.OFFLINE_MODE
    debug_ativo = args.debug

    db = Database()
    api_client = CheapSharkClient(offline_mode=offline)
    bot = DealHuntersBot(database=db, api_client=api_client, seed_on_start=True)

    session_token, boas_vindas = bot.iniciar_sessao(session_token=args.session)
    imprimir_banner(bot.nlu.backend_nlp, offline, session_token)
    print(f"{config.BOT_NAME}: {boas_vindas}\n")

    while True:
        try:
            entrada = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n\n{config.BOT_NAME}: {bot.nlg.despedida()}")
            return 0

        if not entrada:
            continue

        cmd = entrada.lower()

        if cmd in {"/sair", "/exit", "/quit"}:
            print(f"\n{config.BOT_NAME}: {bot.nlg.despedida()}\n")
            return 0

        if cmd == "/debug":
            debug_ativo = not debug_ativo
            status = "ATIVADO" if debug_ativo else "DESATIVADO"
            print(f"\n[Sistema] Modo debug técnico {status}.\n")
            continue

        if cmd == "/reset":
            session_token = str(uuid.uuid4())
            session_token, boas_vindas = bot.iniciar_sessao(session_token=session_token)
            print(f"\n[Sistema] Nova sessão iniciada: {session_token}\n")
            print(f"{config.BOT_NAME}: {boas_vindas}\n")
            continue

        if cmd == "/sessao":
            sessao = db.obter_ou_criar_sessao(session_token=session_token)
            print("\n[SessaoUsuario no SQLite]")
            print(f"  id                 : {sessao.id}")
            print(f"  session_token      : {sessao.session_token}")
            print(f"  estado_atual       : {sessao.estado_atual}")
            print(f"  falhas_consecutivas: {sessao.falhas_consecutivas}")
            print(f"  transbordo_ativo   : {sessao.transbordo_ativo}")
            print(f"  contexto_json      : {sessao.contexto_json}\n")
            continue

        if cmd == "/historico":
            logs = db.listar_logs_sessao(session_token=session_token, limite=12)
            print("\n[ConversaLog — Auditoria LGPD]")
            if not logs:
                print("  (sem registros)")
            for log in logs:
                ts = log.timestamp.strftime("%H:%M:%S")
                resumo_msg = log.mensagem_sanitizada.replace("\n", " ")[:90]
                print(
                    f"  #{log.id} [{ts}] ({log.remetente}) "
                    f"intent={log.intencao_detectada} estado={log.estado_apos_turno} | {resumo_msg}"
                )
            print()
            continue

        if cmd == "/transbordos":
            eventos = db.listar_transbordos(limite=5)
            print("\n[TransbordoHumano — Últimos Protocolos]")
            if not eventos:
                print("  (nenhum transbordo registrado ainda)")
            for ev in eventos:
                print(
                    f"  {ev.protocolo} | motivo={ev.motivo} | prioridade={ev.prioridade} "
                    f"| status={ev.status}"
                )
            print()
            continue

        if cmd == "/ajuda":
            print(f"\n{config.BOT_NAME}: {bot.nlg.ajuda()}\n")
            continue

        resp = bot.responder(entrada, session_token=session_token)
        if debug_ativo:
            imprimir_debug(resp)
        print(f"\n{config.BOT_NAME}: {resp.resposta}\n")

        if resp.intencao == "despedida":
            return 0


if __name__ == "__main__":
    sys.exit(main())
