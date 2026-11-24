"""
Eixo 3 — Persistência Relacional e Estado de Sessão (SQLAlchemy / SQLite).

Implementa:
- Tabela obrigatória `SessaoUsuario` (controle Stateful da máquina de estados e contexto).
- Tabela obrigatória `ConversaLog` (auditoria completa com sanitização LGPD).
- Tabelas corporativas complementares: `PedidoDigital`, `AlertaPreco` e `TransbordoHumano`.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, List, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

import config

Base = declarative_base()

CPF_REGEX = re.compile(r"\b(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})\b")
EMAIL_REGEX = re.compile(
    r"\b([A-Za-z0-9._%+-]{1,64})@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"
)


def utc_now() -> datetime:
    """Retorna timestamp UTC sem tzinfo para compatibilidade limpa com SQLite."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Utilitários de Conformidade com a LGPD (Lei nº 13.709/2018)
# ---------------------------------------------------------------------------


def limpar_cpf(cpf: str) -> str:
    """Remove pontuação e retorna apenas os 11 dígitos do CPF."""
    return re.sub(r"\D", "", cpf or "")


def hash_cpf(cpf: str) -> str:
    """Gera hash SHA-256 do CPF para validação sem armazenar dado pessoal em texto claro."""
    digitos = limpar_cpf(cpf)
    return hashlib.sha256(digitos.encode("utf-8")).hexdigest()


def mascarar_cpf(cpf: str) -> str:
    """Mascara CPF para exibição/log no padrão ***.456.789-** (LGPD)."""
    digitos = limpar_cpf(cpf)
    if len(digitos) != 11:
        return "***.***.***-**"
    return f"***.{digitos[3:6]}.{digitos[6:9]}-**"


def mascarar_email(email: str) -> str:
    """Mascara endereço de e-mail preservando apenas inicial e domínio."""
    if not email or "@" not in email:
        return "***@***.***"
    usuario, dominio = email.split("@", 1)
    prefixo = usuario[0] if usuario else "*"
    return f"{prefixo}***@{dominio}"


def sanitizar_texto_lgpd(texto: str) -> str:
    """Mascara CPFs e e-mails em textos livres antes da persistência em ConversaLog."""
    if not texto:
        return ""
    sanitizado = CPF_REGEX.sub(
        lambda m: f"***.{m.group(2)}.{m.group(3)}-**", texto
    )
    sanitizado = EMAIL_REGEX.sub(
        lambda m: f"{m.group(1)[0]}***@{m.group(2)}", sanitizado
    )
    return sanitizado


def sanitizar_entidades_lgpd(entidades: Dict[str, Any]) -> Dict[str, Any]:
    """Retorna cópia das entidades NLU mascarando dados pessoais sensíveis."""
    copia: Dict[str, Any] = {}
    for chave, valor in (entidades or {}).items():
        if chave == "cpf" and isinstance(valor, str):
            copia[chave] = mascarar_cpf(valor)
        elif chave == "email" and isinstance(valor, str):
            copia[chave] = mascarar_email(valor)
        else:
            copia[chave] = valor
    return copia


# ---------------------------------------------------------------------------
# Modelos ORM (SQLAlchemy)
# ---------------------------------------------------------------------------


class SessaoUsuario(Base):
    """Controla a máquina de estados e o contexto temporário da conversa (Stateful)."""

    __tablename__ = "SessaoUsuario"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_token = Column(String(64), unique=True, nullable=False, index=True)
    usuario_identificador = Column(String(120), nullable=True)
    estado_atual = Column(String(50), nullable=False, default="INICIO")
    contexto_json = Column(Text, nullable=False, default="{}")
    falhas_consecutivas = Column(Integer, nullable=False, default=0)
    consentimento_lgpd = Column(Boolean, nullable=False, default=True)
    transbordo_ativo = Column(Boolean, nullable=False, default=False)
    criado_em = Column(DateTime, nullable=False, default=utc_now)
    atualizado_em = Column(
        DateTime, nullable=False, default=utc_now, onupdate=utc_now
    )

    logs = relationship(
        "ConversaLog",
        back_populates="sessao",
        cascade="all, delete-orphan",
        order_by="ConversaLog.id",
    )
    alertas = relationship(
        "AlertaPreco",
        back_populates="sessao",
        cascade="all, delete-orphan",
    )
    transbordos = relationship(
        "TransbordoHumano",
        back_populates="sessao",
        cascade="all, delete-orphan",
    )

    def get_contexto(self) -> Dict[str, Any]:
        try:
            return json.loads(self.contexto_json or "{}")
        except json.JSONDecodeError:
            return {}

    def set_contexto(self, dados: Dict[str, Any]) -> None:
        self.contexto_json = json.dumps(dados, ensure_ascii=False)


class ConversaLog(Base):
    """Auditoria completa de turnos da conversa em conformidade com a LGPD."""

    __tablename__ = "ConversaLog"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sessao_id = Column(
        Integer, ForeignKey("SessaoUsuario.id"), nullable=False, index=True
    )
    remetente = Column(String(20), nullable=False)  # usuario | bot | sistema
    mensagem_sanitizada = Column(Text, nullable=False)
    intencao_detectada = Column(String(60), nullable=True)
    confianca_nlu = Column(Float, nullable=True)
    entidades_json = Column(Text, nullable=True, default="{}")
    sentimento = Column(String(30), nullable=True)
    estado_apos_turno = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False, default=utc_now)
    anonimizado_lgpd = Column(Boolean, nullable=False, default=False)

    sessao = relationship("SessaoUsuario", back_populates="logs")


class PedidoDigital(Base):
    """Base corporativa de pedidos de chaves digitais para atendimento pós-venda."""

    __tablename__ = "PedidoDigital"

    id = Column(Integer, primary_key=True, autoincrement=True)
    codigo_pedido = Column(String(20), unique=True, nullable=False, index=True)
    cliente_nome = Column(String(100), nullable=False)
    cpf_hash = Column(String(64), nullable=False, index=True)
    cpf_mascarado = Column(String(20), nullable=False)
    email_mascarado = Column(String(120), nullable=False)
    jogo_titulo = Column(String(150), nullable=False)
    plataforma = Column(String(50), nullable=False)
    valor_brl = Column(Float, nullable=False)
    status = Column(String(40), nullable=False)
    chave_ativacao = Column(String(60), nullable=True)
    data_compra = Column(DateTime, nullable=False, default=utc_now)


class AlertaPreco(Base):
    """Alertas de preço cadastrados pelos usuários durante o atendimento."""

    __tablename__ = "AlertaPreco"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sessao_id = Column(
        Integer, ForeignKey("SessaoUsuario.id"), nullable=True, index=True
    )
    jogo_titulo = Column(String(150), nullable=False)
    preco_alvo_brl = Column(Float, nullable=False)
    contato_mascarado = Column(String(120), nullable=True)
    ativo = Column(Boolean, nullable=False, default=True)
    criado_em = Column(DateTime, nullable=False, default=utc_now)

    sessao = relationship("SessaoUsuario", back_populates="alertas")


class TransbordoHumano(Base):
    """Registro de auditoria de escalonamentos para atendentes humanos (Eixo 4)."""

    __tablename__ = "TransbordoHumano"

    id = Column(Integer, primary_key=True, autoincrement=True)
    protocolo = Column(String(30), unique=True, nullable=False, index=True)
    sessao_id = Column(
        Integer, ForeignKey("SessaoUsuario.id"), nullable=False, index=True
    )
    motivo = Column(String(60), nullable=False)
    prioridade = Column(String(20), nullable=False, default="NORMAL")
    resumo_contexto = Column(Text, nullable=False)
    status = Column(String(30), nullable=False, default="AGUARDANDO_ATENDENTE")
    criado_em = Column(DateTime, nullable=False, default=utc_now)

    sessao = relationship("SessaoUsuario", back_populates="transbordos")


# ---------------------------------------------------------------------------
# Classe Database (Camada de Acesso a Dados Desacoplada)
# ---------------------------------------------------------------------------


class Database:
    """Gerencia engine, sessões ORM e operações de persistência no SQLite."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        self.database_url = database_url or config.DATABASE_URL
        connect_args = (
            {"check_same_thread": False}
            if self.database_url.startswith("sqlite")
            else {}
        )
        self.engine = create_engine(
            self.database_url,
            echo=False,
            future=True,
            connect_args=connect_args,
        )
        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
            future=True,
        )
        self.criar_tabelas()

    def criar_tabelas(self) -> None:
        """Cria todas as tabelas relacionais caso ainda não existam."""
        Base.metadata.create_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Iterator[Session]:
        """Context manager seguro para transações SQLAlchemy."""
        db: Session = self.SessionLocal()
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ---------------------- Gestão de SessaoUsuario ----------------------

    def obter_ou_criar_sessao(
        self,
        session_token: Optional[str] = None,
        usuario_identificador: Optional[str] = None,
    ) -> SessaoUsuario:
        token = session_token or str(uuid.uuid4())
        with self.get_session() as db:
            sessao = (
                db.query(SessaoUsuario)
                .filter(SessaoUsuario.session_token == token)
                .first()
            )
            if sessao is None:
                sessao = SessaoUsuario(
                    session_token=token,
                    usuario_identificador=usuario_identificador,
                    estado_atual="INICIO",
                    contexto_json="{}",
                    falhas_consecutivas=0,
                    consentimento_lgpd=True,
                    transbordo_ativo=False,
                )
                db.add(sessao)
                db.flush()
            return sessao

    def atualizar_sessao(
        self,
        session_token: str,
        *,
        estado_atual: Optional[str] = None,
        contexto: Optional[Dict[str, Any]] = None,
        falhas_consecutivas: Optional[int] = None,
        transbordo_ativo: Optional[bool] = None,
        usuario_identificador: Optional[str] = None,
    ) -> SessaoUsuario:
        with self.get_session() as db:
            sessao = (
                db.query(SessaoUsuario)
                .filter(SessaoUsuario.session_token == session_token)
                .first()
            )
            if sessao is None:
                raise ValueError(f"Sessão não encontrada: {session_token}")

            if estado_atual is not None:
                sessao.estado_atual = estado_atual
            if contexto is not None:
                sessao.set_contexto(contexto)
            if falhas_consecutivas is not None:
                sessao.falhas_consecutivas = falhas_consecutivas
            if transbordo_ativo is not None:
                sessao.transbordo_ativo = transbordo_ativo
            if usuario_identificador is not None:
                sessao.usuario_identificador = usuario_identificador
            sessao.atualizado_em = utc_now()
            db.flush()
            return sessao

    def resetar_sessao(self, session_token: str) -> SessaoUsuario:
        return self.atualizar_sessao(
            session_token,
            estado_atual="INICIO",
            contexto={},
            falhas_consecutivas=0,
            transbordo_ativo=False,
        )

    # ---------------------- Auditoria ConversaLog (LGPD) ----------------------

    def registrar_log(
        self,
        sessao_id: int,
        remetente: str,
        mensagem: str,
        *,
        intencao_detectada: Optional[str] = None,
        confianca_nlu: Optional[float] = None,
        entidades: Optional[Dict[str, Any]] = None,
        sentimento: Optional[str] = None,
        estado_apos_turno: Optional[str] = None,
    ) -> ConversaLog:
        mensagem_segura = sanitizar_texto_lgpd(mensagem)
        entidades_seguras = sanitizar_entidades_lgpd(entidades or {})
        with self.get_session() as db:
            log = ConversaLog(
                sessao_id=sessao_id,
                remetente=remetente,
                mensagem_sanitizada=mensagem_segura,
                intencao_detectada=intencao_detectada,
                confianca_nlu=confianca_nlu,
                entidades_json=json.dumps(entidades_seguras, ensure_ascii=False),
                sentimento=sentimento,
                estado_apos_turno=estado_apos_turno,
                anonimizado_lgpd=False,
            )
            db.add(log)
            db.flush()
            return log

    def listar_logs_sessao(
        self, session_token: str, limite: int = 20
    ) -> List[ConversaLog]:
        with self.get_session() as db:
            sessao = (
                db.query(SessaoUsuario)
                .filter(SessaoUsuario.session_token == session_token)
                .first()
            )
            if not sessao:
                return []
            logs = (
                db.query(ConversaLog)
                .filter(ConversaLog.sessao_id == sessao.id)
                .order_by(ConversaLog.id.desc())
                .limit(limite)
                .all()
            )
            return list(reversed(logs))

    def anonimizar_sessao_lgpd(self, session_token: str) -> int:
        """
        Aplica direito à anonimização (Art. 18, LGPD) nos registros de ConversaLog
        da sessão informada, preservando apenas metadados estatísticos.
        """
        with self.get_session() as db:
            sessao = (
                db.query(SessaoUsuario)
                .filter(SessaoUsuario.session_token == session_token)
                .first()
            )
            if not sessao:
                return 0

            sessao.usuario_identificador = "[ANONIMIZADO_LGPD]"
            logs = (
                db.query(ConversaLog)
                .filter(ConversaLog.sessao_id == sessao.id)
                .all()
            )
            count = 0
            for log in logs:
                if not log.anonimizado_lgpd:
                    log.mensagem_sanitizada = "[CONTEUDO ANONIMIZADO - LGPD]"
                    log.entidades_json = "{}"
                    log.anonimizado_lgpd = True
                    count += 1
            return count

    # ---------------------- Pedidos Digitais (Pós-venda) ----------------------

    def buscar_pedido_por_codigo(
        self, codigo_pedido: str
    ) -> Optional[PedidoDigital]:
        codigo_norm = (codigo_pedido or "").strip().upper()
        if not codigo_norm.startswith("DH-") and codigo_norm.isdigit():
            codigo_norm = f"DH-{codigo_norm}"
        with self.get_session() as db:
            return (
                db.query(PedidoDigital)
                .filter(PedidoDigital.codigo_pedido == codigo_norm)
                .first()
            )

    def validar_cpf_pedido(self, codigo_pedido: str, cpf: str) -> Optional[PedidoDigital]:
        pedido = self.buscar_pedido_por_codigo(codigo_pedido)
        if not pedido:
            return None
        if pedido.cpf_hash == hash_cpf(cpf):
            return pedido
        return None

    # ---------------------- Alertas de Preço ----------------------

    def criar_alerta_preco(
        self,
        sessao_id: Optional[int],
        jogo_titulo: str,
        preco_alvo_brl: float,
        contato: Optional[str] = None,
    ) -> AlertaPreco:
        contato_seguro = mascarar_email(contato) if contato else None
        with self.get_session() as db:
            alerta = AlertaPreco(
                sessao_id=sessao_id,
                jogo_titulo=jogo_titulo.strip(),
                preco_alvo_brl=round(float(preco_alvo_brl), 2),
                contato_mascarado=contato_seguro,
                ativo=True,
            )
            db.add(alerta)
            db.flush()
            return alerta

    def listar_alertas_sessao(self, sessao_id: int) -> List[AlertaPreco]:
        with self.get_session() as db:
            return (
                db.query(AlertaPreco)
                .filter(
                    AlertaPreco.sessao_id == sessao_id,
                    AlertaPreco.ativo.is_(True),
                )
                .order_by(AlertaPreco.criado_em.desc())
                .all()
            )

    # ---------------------- Transbordo Humano (Eixo 4) ----------------------

    def registrar_transbordo(
        self,
        sessao_id: int,
        motivo: str,
        prioridade: str,
        resumo_contexto: str,
    ) -> TransbordoHumano:
        sufixo = uuid.uuid4().hex[:6].upper()
        data_tag = utc_now().strftime("%Y%m%d")
        protocolo = f"DH-ATD-{data_tag}-{sufixo}"
        with self.get_session() as db:
            evento = TransbordoHumano(
                protocolo=protocolo,
                sessao_id=sessao_id,
                motivo=motivo,
                prioridade=prioridade,
                resumo_contexto=sanitizar_texto_lgpd(resumo_contexto),
                status="AGUARDANDO_ATENDENTE",
            )
            db.add(evento)
            db.flush()
            return evento

    def listar_transbordos(self, limite: int = 10) -> List[TransbordoHumano]:
        with self.get_session() as db:
            return (
                db.query(TransbordoHumano)
                .order_by(TransbordoHumano.id.desc())
                .limit(limite)
                .all()
            )

    # ---------------------- Seed / Dados Pré-Populados ----------------------

    def popular_dados_teste(self) -> Dict[str, int]:
        """
        Pré-popula o banco SQLite de forma idempotente com pedidos digitais,
        um alerta de exemplo e uma sessão histórica de auditoria para testes.
        """
        agora = utc_now()
        pedidos_seed = [
            {
                "codigo_pedido": "DH-1001",
                "cliente_nome": "Lucas Almeida",
                "cpf": "111.222.333-44",
                "email": "lucas.almeida@exemplo.com",
                "jogo_titulo": "The Witcher 3: Wild Hunt",
                "plataforma": "Steam",
                "valor_brl": 46.34,
                "status": "ENTREGUE",
                "chave_ativacao": "DHW3-8821-STEAM-9941",
                "data_compra": agora - timedelta(days=5),
            },
            {
                "codigo_pedido": "DH-1002",
                "cliente_nome": "Mariana Costa",
                "cpf": "555.666.777-88",
                "email": "mariana.costa@exemplo.com",
                "jogo_titulo": "Elden Ring",
                "plataforma": "Steam",
                "valor_brl": 226.14,
                "status": "CHAVE_PENDENTE",
                "chave_ativacao": None,
                "data_compra": agora - timedelta(hours=3),
            },
            {
                "codigo_pedido": "DH-1003",
                "cliente_nome": "Rafael Souza",
                "cpf": "999.888.777-66",
                "email": "rafael.souza@exemplo.com",
                "jogo_titulo": "Hollow Knight",
                "plataforma": "GOG",
                "valor_brl": 43.44,
                "status": "AGUARDANDO_PAGAMENTO",
                "chave_ativacao": None,
                "data_compra": agora - timedelta(days=1),
            },
            {
                "codigo_pedido": "DH-1004",
                "cliente_nome": "Lucas Almeida",
                "cpf": "111.222.333-44",
                "email": "lucas.almeida@exemplo.com",
                "jogo_titulo": "Cyberpunk 2077",
                "plataforma": "GOG",
                "valor_brl": 173.94,
                "status": "REEMBOLSO_EM_ANALISE",
                "chave_ativacao": None,
                "data_compra": agora - timedelta(days=2),
            },
        ]

        criados = {"pedidos": 0, "sessoes": 0, "alertas": 0}

        with self.get_session() as db:
            for item in pedidos_seed:
                existente = (
                    db.query(PedidoDigital)
                    .filter(PedidoDigital.codigo_pedido == item["codigo_pedido"])
                    .first()
                )
                if not existente:
                    db.add(
                        PedidoDigital(
                            codigo_pedido=item["codigo_pedido"],
                            cliente_nome=item["cliente_nome"],
                            cpf_hash=hash_cpf(item["cpf"]),
                            cpf_mascarado=mascarar_cpf(item["cpf"]),
                            email_mascarado=mascarar_email(item["email"]),
                            jogo_titulo=item["jogo_titulo"],
                            plataforma=item["plataforma"],
                            valor_brl=item["valor_brl"],
                            status=item["status"],
                            chave_ativacao=item["chave_ativacao"],
                            data_compra=item["data_compra"],
                        )
                    )
                    criados["pedidos"] += 1

            # Sessão demonstrativa pré-populada para evidência de auditoria no SQLite
            demo_token = "sessao-demo-auditoria-001"
            sessao_demo = (
                db.query(SessaoUsuario)
                .filter(SessaoUsuario.session_token == demo_token)
                .first()
            )
            if not sessao_demo:
                sessao_demo = SessaoUsuario(
                    session_token=demo_token,
                    usuario_identificador="Lucas Almeida",
                    estado_atual="EXIBINDO_OFERTAS",
                    contexto_json=json.dumps(
                        {"ultimo_jogo": "Hollow Knight"}, ensure_ascii=False
                    ),
                    falhas_consecutivas=0,
                    consentimento_lgpd=True,
                    transbordo_ativo=False,
                )
                db.add(sessao_demo)
                db.flush()
                criados["sessoes"] += 1

                db.add(
                    ConversaLog(
                        sessao_id=sessao_demo.id,
                        remetente="usuario",
                        mensagem_sanitizada="Quanto custa Hollow Knight hoje?",
                        intencao_detectada="consultar_preco",
                        confianca_nlu=0.92,
                        entidades_json=json.dumps(
                            {"jogo": "Hollow Knight"}, ensure_ascii=False
                        ),
                        sentimento="NEUTRO",
                        estado_apos_turno="EXIBINDO_OFERTAS",
                    )
                )
                db.add(
                    ConversaLog(
                        sessao_id=sessao_demo.id,
                        remetente="bot",
                        mensagem_sanitizada=(
                            "Encontrei ofertas para Hollow Knight a partir de "
                            "US$ 7.49 (aprox. R$ 43,44) na Steam."
                        ),
                        intencao_detectada="consultar_preco",
                        confianca_nlu=0.92,
                        entidades_json="{}",
                        sentimento="NEUTRO",
                        estado_apos_turno="EXIBINDO_OFERTAS",
                    )
                )
                db.add(
                    AlertaPreco(
                        sessao_id=sessao_demo.id,
                        jogo_titulo="Elden Ring",
                        preco_alvo_brl=150.00,
                        contato_mascarado="l***@exemplo.com",
                        ativo=True,
                    )
                )
                criados["alertas"] += 1

        return criados
