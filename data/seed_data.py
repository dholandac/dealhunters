"""
Script utilitário para inicializar e pré-popular o banco SQLite (dealhunters.db)
com pedidos digitais fictícios, alerta e log de auditoria para testes e Live Demo.

Uso:
    python data/seed_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import Database


def main() -> None:
    db = Database()
    resumo = db.popular_dados_teste()
    print(f"[Deal Hunters] Banco inicializado em: {db.database_url}")
    print(
        "[Deal Hunters] Registros inseridos -> "
        f"Pedidos: {resumo['pedidos']} | "
        f"Sessões demo: {resumo['sessoes']} | "
        f"Alertas: {resumo['alertas']}"
    )
    print("\nPedidos disponíveis para teste no atendimento pós-venda:")
    print("  - DH-1001 | CPF: 111.222.333-44 | The Witcher 3: Wild Hunt (ENTREGUE + Key)")
    print("  - DH-1002 | CPF: 555.666.777-88 | Elden Ring (CHAVE_PENDENTE)")
    print("  - DH-1003 | CPF: 999.888.777-66 | Hollow Knight (AGUARDANDO_PAGAMENTO)")
    print("  - DH-1004 | CPF: 111.222.333-44 | Cyberpunk 2077 (REEMBOLSO_EM_ANALISE)")


if __name__ == "__main__":
    main()
