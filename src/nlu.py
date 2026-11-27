"""
Eixo 1 — Processamento de Linguagem Natural (NLU/NLP) com spaCy e NLTK.

Funcionalidades:
- Pré-processamento estrutural: normalização de gírias/ruídos ortográficos,
  correção aproximada (fuzzy) de vocabulário de domínio, tokenização, remoção
  inteligente de stopwords e lematização via spaCy (pt_core_news_sm), com
  fallback estrutural via spaCy blank('pt') + NLTK RSLPStemmer caso o modelo
  estatístico ainda não tenha sido baixado.
- Reconhecimento de Entidades Nomeadas (NER) híbrido:
  EntityRuler customizado do spaCy + NER nativo + regras sintáticas para
  `jogo`, `loja`, `valor_brl`, `codigo_pedido`, `cpf`, `email` e `opcao_numero`.
- Classificação de Intenções baseada em vetorização TF-IDF + Similaridade de
  Cosseno sobre lemas (com mascaramento de entidades detectadas).
- Detecção de Sentimento e Urgência para acionamento do transbordo humano (Eixo 4).
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import spacy
from spacy.language import Language

import config

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import RSLPStemmer

    NLTK_AVAILABLE = True
except Exception:
    nltk_stopwords = None
    RSLPStemmer = None
    NLTK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Utilitários de Texto e Vocabulário de Domínio
# ---------------------------------------------------------------------------


def remover_acentos(texto: str) -> str:
    """Remove diacríticos preservando letras ASCII base."""
    normalizado = unicodedata.normalize("NFKD", texto or "")
    return "".join(ch for ch in normalizado if not unicodedata.combining(ch))


# Stopwords semânticas preservadas para não perder negações, valores ou perguntas
STOPWORDS_PRESERVAR = {
    "nao",
    "nunca",
    "nem",
    "sem",
    "jamais",
    "quanto",
    "qual",
    "quais",
    "como",
    "ate",
    "mais",
    "menos",
    "bom",
    "boa",
    "melhor",
    "primeiro",
    "primeira",
    "segundo",
    "segunda",
    "terceiro",
    "terceira",
    "quarto",
    "quarta",
    "quinto",
    "quinta",
    "link",
}

STOPWORDS_PT_BASICAS = {
    "a",
    "ao",
    "aos",
    "as",
    "com",
    "da",
    "das",
    "de",
    "do",
    "dos",
    "e",
    "em",
    "essa",
    "esse",
    "esta",
    "este",
    "eu",
    "me",
    "meu",
    "minha",
    "na",
    "nas",
    "no",
    "nos",
    "o",
    "os",
    "ou",
    "para",
    "pela",
    "pelo",
    "por",
    "pra",
    "pro",
    "que",
    "se",
    "seu",
    "sua",
    "ta",
    "te",
    "to",
    "um",
    "uma",
    "uns",
    "umas",
    "voce",
}

# Normalização de gírias, abreviações e erros ortográficos comuns no pt-BR gamer
NORMALIZACAO_TOKENS: Dict[str, str] = {
    "qnt": "quanto",
    "qnto": "quanto",
    "qt": "quanto",
    "qto": "quanto",
    "qunto": "quanto",
    "quntu": "quanto",
    "ta": "esta",
    "tah": "esta",
    "tá": "esta",
    "to": "estou",
    "tô": "estou",
    "vc": "voce",
    "vcs": "voces",
    "kero": "quero",
    "qro": "quero",
    "keria": "queria",
    "qria": "queria",
    "sabe": "saber",
    "preso": "preco",
    "preço": "preco",
    "precos": "preco",
    "preços": "preco",
    "custa": "custar",
    "custando": "custar",
    "promo": "promocao",
    "promos": "promocao",
    "promocoes": "promocao",
    "promossao": "promocao",
    "promosao": "promocao",
    "ofrta": "oferta",
    "ofertas": "oferta",
    "baratinho": "barato",
    "baratin": "barato",
    "massa": "bom",
    "top": "bom",
    "conto": "reais",
    "contos": "reais",
    "pila": "reais",
    "pilas": "reais",
    "mangos": "reais",
    "mango": "reais",
    "real": "reais",
    "pedidu": "pedido",
    "pedidos": "pedido",
    "key": "chave",
    "keys": "chave",
    "chve": "chave",
    "chaves": "chave",
    "codigo": "codigo",
    "cod": "codigo",
    "reembolco": "reembolso",
    "reemboso": "reembolso",
    "reembolssar": "reembolso",
    "estorno": "reembolso",
    "devolucao": "reembolso",
    "extorno": "reembolso",
    "cade": "onde",
    "kd": "onde",
    "blz": "beleza",
    "obg": "obrigado",
    "obgd": "obrigado",
    "vlw": "obrigado",
    "valeu": "obrigado",
    "flw": "tchau",
    "falou": "tchau",
    "eai": "ola",
    "eae": "ola",
    "opa": "ola",
    "salve": "ola",
    "pfv": "favor",
    "pf": "favor",
    "plz": "favor",
    "atend": "atendente",
    "suport": "suporte",
    "human": "humano",
    "pessimo": "pessimo",
    "orivel": "horrivel",
    "absurd": "absurdo",
    "urgnt": "urgente",
    "aviza": "avisar",
    "avisa": "avisar",
    "alerta": "alerta",
    "notifica": "avisar",
    "opc": "opcao",
    "opcoes": "opcao",
    "num": "numero",
    "n": "nao",
    "nn": "nao",
    "ñ": "nao",
}

# Apelidos populares de jogos (normalizados antes do NER)
APELIDOS_JOGOS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:o\s+)?bruxeiro(?:\s*3)?\b", re.IGNORECASE), "The Witcher 3"),
    (re.compile(r"\bwitcher\s*3\b", re.IGNORECASE), "The Witcher 3"),
    (re.compile(r"\bhollow\s*nigth\b|\bholow\s*knight\b", re.IGNORECASE), "Hollow Knight"),
    (re.compile(r"\belden\s*ringue\b|\beldem\s*ring\b", re.IGNORECASE), "Elden Ring"),
    (re.compile(r"\bgta\s*(?:5|v)\b", re.IGNORECASE), "Grand Theft Auto V"),
    (re.compile(r"\brdr\s*2\b|\bred\s+dead\s+2\b", re.IGNORECASE), "Red Dead Redemption 2"),
    (re.compile(r"\bre\s*4\b", re.IGNORECASE), "Resident Evil 4"),
    (re.compile(r"\bbg\s*3\b|\bbaldurs\s+gate\s*3\b", re.IGNORECASE), "Baldur's Gate 3"),
    (re.compile(r"\bgow\b", re.IGNORECASE), "God of War"),
    (re.compile(r"\bcp\s*2077\b|\bciberpunk(?:\s*2077)?\b", re.IGNORECASE), "Cyberpunk 2077"),
    (re.compile(r"\bds\s*3\b", re.IGNORECASE), "Dark Souls III"),
    (re.compile(r"\bstardew\b(?!\s+valley)", re.IGNORECASE), "Stardew Valley"),
]

# Jogos conhecidos para o EntityRuler do spaCy (além da extração dinâmica de qualquer título)
CATALOGO_JOGOS_NER: List[str] = [
    "The Witcher 3: Wild Hunt",
    "The Witcher 3",
    "Hollow Knight",
    "Elden Ring",
    "Cyberpunk 2077",
    "Hades",
    "Hades II",
    "Stardew Valley",
    "Red Dead Redemption 2",
    "Resident Evil 4",
    "Resident Evil 2",
    "God of War",
    "Baldur's Gate 3",
    "Grand Theft Auto V",
    "Batman: Arkham Knight",
    "Batman: Arkham Asylum",
    "Dark Souls III",
    "Dark Souls",
    "Sekiro",
    "Skyrim",
    "Portal 2",
    "Portal",
    "Terraria",
    "Celeste",
    "Cuphead",
    "Doom Eternal",
    "Fallout 4",
    "Fallout: New Vegas",
    "Persona 5",
    "Dead Cells",
    "Subnautica",
    "Valheim",
    "Factorio",
    "Palworld",
    "Hogwarts Legacy",
    "Monster Hunter: World",
    "BioShock Infinite",
    "Mass Effect",
    "Half-Life 2",
    "Left 4 Dead 2",
    "Street Fighter 6",
    "Mortal Kombat 1",
    "Diablo IV",
    "The Last of Us",
]

LOJAS_NER: Dict[str, str] = {
    "steam": "Steam",
    "gog": "GOG",
    "epic games store": "Epic Games Store",
    "epic games": "Epic Games Store",
    "epic": "Epic Games Store",
    "humble store": "Humble Store",
    "humble bundle": "Humble Store",
    "humble": "Humble Store",
    "fanatical": "Fanatical",
    "green man gaming": "GreenManGaming",
    "greenmangaming": "GreenManGaming",
    "gmg": "GreenManGaming",
    "gamersgate": "GamersGate",
    "ubisoft": "Ubisoft Connect",
    "uplay": "Ubisoft Connect",
    "gamesplanet": "Gamesplanet",
    "nuuvem": "Nuuvem",
}

TERMOS_NAO_JOGO = {
    "um",
    "uma",
    "o",
    "a",
    "os",
    "as",
    "esse",
    "este",
    "essa",
    "esta",
    "isso",
    "ele",
    "ela",
    "jogo",
    "jogos",
    "game",
    "games",
    "preco",
    "valor",
    "promocao",
    "oferta",
    "desconto",
    "barato",
    "baratos",
    "bom",
    "bons",
    "boa",
    "boas",
    "legal",
    "hoje",
    "agora",
    "aqui",
    "qual",
    "quais",
    "quanto",
    "quando",
    "como",
    "onde",
    "quem",
    "baixar",
    "cair",
    "algum",
    "alguma",
    "meu",
    "minha",
    "pedido",
    "chave",
    "compra",
    "reembolso",
    "alerta",
    "atendente",
    "humano",
    "suporte",
    "dados",
    "lgpd",
    "privacidade",
    "opcao",
    "link",
    "mais",
    "menos",
    "detalhes",
    "informacoes",
    "favor",
}

# Palavras válidas comuns que nunca devem ser alteradas pela correção aproximada (difflib)
PALAVRAS_COMUNS_NAO_CORRIGIR = {
    "quando",
    "quanto",
    "onde",
    "porque",
    "como",
    "quem",
    "quais",
    "qual",
    "baixar",
    "cair",
    "subir",
    "pagar",
    "paguei",
    "comprei",
    "comprar",
    "quero",
    "queria",
    "preciso",
    "gostaria",
    "posso",
    "pode",
    "saber",
    "mostrar",
    "falar",
    "dizer",
    "melhor",
    "maior",
    "menor",
    "barata",
    "baratas",
    "baratos",
    "gratis",
    "gratuito",
    "ontem",
    "hoje",
    "amanha",
    "agora",
    "depois",
    "antes",
    "muito",
    "pouco",
    "ainda",
    "tambem",
    "receita",
    "partida",
    "ganhou",
}

# Vocabulário canônico para correção ortográfica aproximada (difflib)
VOCABULARIO_DOMINIO = sorted(
    {
        "quanto",
        "custar",
        "preco",
        "valor",
        "jogo",
        "promocao",
        "oferta",
        "desconto",
        "barato",
        "orcamento",
        "reais",
        "loja",
        "comparar",
        "melhor",
        "alerta",
        "avisar",
        "baixar",
        "notificar",
        "pedido",
        "rastrear",
        "status",
        "chave",
        "ativacao",
        "entrega",
        "compra",
        "reembolso",
        "cancelar",
        "devolver",
        "atendente",
        "humano",
        "suporte",
        "pessoa",
        "transferir",
        "ajuda",
        "opcoes",
        "opcao",
        "detalhes",
        "link",
        "privacidade",
        "dados",
        "lgpd",
        "apagar",
        "anonimizar",
        "excluir",
        "esquecimento",
        "obrigado",
        "tchau",
        "encerrar",
        "urgente",
        "absurdo",
        "pessimo",
        "horrivel",
        "invalida",
        "funciona",
        "procon",
        "fraude",
        "golpe",
    }
)


# ---------------------------------------------------------------------------
# Base de Treinamento de Intenções (usada pelo classificador TF-IDF + Cosseno)
# Entidades detectadas são substituídas por marcadores (__jogo__, __valor__, etc.)
# ---------------------------------------------------------------------------

EXEMPLOS_INTENCOES: Dict[str, List[str]] = {
    "saudacao": [
        "oi",
        "ola",
        "bom dia",
        "boa tarde",
        "boa noite",
        "ola hunter tudo bem",
        "oi preciso de ajuda",
        "ola quero atendimento",
    ],
    "despedida": [
        "tchau",
        "ate mais",
        "ate logo",
        "obrigado tchau",
        "muito obrigado era so isso",
        "encerrar atendimento",
        "pode finalizar obrigado",
        "sair do chat",
    ],
    "ajuda": [
        "ajuda",
        "o que voce faz",
        "como funciona",
        "quais opcoes voce tem",
        "menu principal",
        "me mostra o que posso perguntar",
        "quais servicos voces oferecem",
    ],
    "consultar_preco": [
        "quanto custar __jogo__",
        "qual preco __jogo__",
        "quanto esta __jogo__",
        "quanto esta __jogo__ na __loja__",
        "quero saber preco __jogo__",
        "preco __jogo__ hoje",
        "buscar preco __jogo__",
        "comparar preco __jogo__ em loja",
        "tem __jogo__ disponivel",
        "quanto custar um jogo",
        "quero ver preco de um jogo",
        "pesquisar preco de jogo",
        "qual valor __jogo__",
        "me fala preco __jogo__",
    ],
    "buscar_promocoes": [
        "quais jogos estao em promocao hoje",
        "tem promocao hoje",
        "quero ver oferta de jogos",
        "mostrar desconto em jogos",
        "jogos baratos ate __valor__",
        "promocao ate __valor__",
        "promocao boa ate __valor__ na __loja__",
        "oferta ate __valor__ na __loja__",
        "desconto na __loja__",
        "o que esta em promocao na __loja__",
        "filtrar oferta ate __valor__",
        "so na __loja__",
        "melhores promocoes de hoje",
        "tem oferta boa barata",
    ],
    "selecionar_opcao": [
        "quero opcao __opcao__",
        "me mostra detalhe da opcao __opcao__",
        "me manda link da opcao __opcao__",
        "quero saber mais da __opcao__",
        "detalhes da opcao __opcao__",
        "escolho a __opcao__",
        "ver link do __opcao__",
        "opcao __opcao__",
    ],
    "criar_alerta": [
        "me avisar quando __jogo__ baixar para __valor__",
        "criar alerta de preco para __jogo__ ate __valor__",
        "quero alerta para __jogo__ por __valor__",
        "avisar quando baixar para __valor__",
        "criar alerta de preco para esse jogo",
        "me notificar quando cair para __valor__",
        "acompanhar preco de __jogo__",
        "quero criar um alerta de preco",
        "listar meus alertas de preco",
    ],
    "consultar_pedido": [
        "consultar pedido __pedido__",
        "quero ver meu pedido __pedido__",
        "status do pedido __pedido__",
        "onde esta minha chave do pedido __pedido__",
        "ver chave de ativacao do pedido __pedido__",
        "consultar minha compra",
        "onde esta minha chave",
        "quero rastrear meu pedido",
        "acompanhar entrega da chave digital",
        "meu pedido __pedido__ cpf __cpf__",
        "validar pedido com cpf __cpf__",
    ],
    "solicitar_reembolso": [
        "quero pedir reembolso",
        "solicitar reembolso do pedido __pedido__",
        "cancelar compra e devolver dinheiro",
        "quero estorno do meu pedido",
        "minha chave nao funcionou quero reembolso",
        "como faco para cancelar pedido __pedido__",
    ],
    "falar_atendente": [
        "quero falar com atendente humano",
        "me transfere para um atendente",
        "preciso falar com uma pessoa de verdade",
        "chamar suporte humano",
        "transferir para especialista",
        "nao quero falar com robo",
    ],
    "privacidade_lgpd": [
        "como voces tratam meus dados lgpd",
        "politica de privacidade lgpd",
        "meus dados estao seguros lgpd",
        "o que voces guardam do meu cpf",
        "informacoes sobre privacidade",
    ],
    "anonimizar_lgpd": [
        "apagar meus dados da conversa",
        "anonimizar meu historico lgpd",
        "excluir meus dados pessoais",
        "quero direito ao esquecimento lgpd",
        "remover meus registros do banco",
    ],
}

# Léxico de sentimento negativo e gatilhos de urgência/insatisfação extrema (Eixo 4)
TERMOS_CRITICOS_URGENCIA = {
    "absurdo": 0.45,
    "lixo": 0.50,
    "pessimo": 0.45,
    "horrivel": 0.45,
    "raiva": 0.40,
    "furioso": 0.50,
    "indignado": 0.45,
    "golpe": 0.65,
    "fraude": 0.65,
    "roubo": 0.60,
    "procon": 0.70,
    "processar": 0.65,
    "advogado": 0.55,
    "urgente": 0.40,
    "urgencia": 0.40,
    "invalida": 0.45,
    "invalido": 0.40,
    "enganacao": 0.55,
    "vergonha": 0.45,
    "ridiculo": 0.45,
    "inaceitavel": 0.50,
    "reclame": 0.45,
    "cancelar": 0.20,
    "reembolso": 0.25,
}

FRASES_CRITICAS_URGENCIA = [
    ("nao funciona", 0.45),
    ("nao funcionou", 0.45),
    ("nao ativou", 0.45),
    ("nao ativa", 0.45),
    ("nao recebi", 0.50),
    ("nao chegou", 0.45),
    ("cobrado duas vezes", 0.70),
    ("cobranca duplicada", 0.70),
    ("paguei e nao", 0.60),
    ("chave ja usada", 0.65),
    ("chave invalida", 0.65),
    ("conta invadida", 0.75),
    ("quero meu dinheiro", 0.55),
]

TERMOS_POSITIVOS = {
    "obrigado",
    "otimo",
    "excelente",
    "perfeito",
    "maravilha",
    "bom",
    "adorei",
    "gostei",
    "ajudou",
    "beleza",
}


# ---------------------------------------------------------------------------
# Estruturas de Saída do NLU
# ---------------------------------------------------------------------------


@dataclass
class NLUResult:
    """Resultado completo do pipeline NLP/NLU para uma mensagem do usuário."""

    texto_original: str
    texto_normalizado: str
    tokens: List[str]
    tokens_sem_stopwords: List[str]
    lemas: List[str]
    intencao: str
    confianca: float
    entidades: Dict[str, Any] = field(default_factory=dict)
    sentimento: str = "NEUTRO"  # POSITIVO | NEUTRO | NEGATIVO | CRITICO_URGENTE
    score_insatisfacao: float = 0.0
    backend_nlp: str = "spacy"
    ranking_intencoes: List[Tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classe NLU (Módulo de Processamento de Linguagem Natural Desacoplado)
# ---------------------------------------------------------------------------


class NLU:
    """
    Motor de NLU/NLP do Deal Hunters baseado em spaCy (+ suporte NLTK).
    Realiza pré-processamento estrutural, NER, classificação de intenções e
    análise de sentimento/urgência.
    """

    def __init__(
        self,
        model_name: str = config.SPACY_MODEL_NAME,
        confidence_threshold: float = config.NLU_CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.backend_nlp = "spacy"
        self._nltk_stemmer = None
        self._stopwords = set(STOPWORDS_PT_BASICAS)

        self._carregar_stopwords_nltk()
        self.nlp: Language = self._inicializar_spacy()
        self._configurar_entity_ruler()

        # Índices TF-IDF pré-calculados a partir da base de treinamento
        self._idf: Dict[str, float] = {}
        self._max_idf: float = 3.0
        self._vetores_treinamento: List[Tuple[str, Dict[str, float], float]] = []
        self._treinar_classificador()

    # ------------------------------------------------------------------
    # Inicialização do Pipeline spaCy / NLTK
    # ------------------------------------------------------------------

    def _carregar_stopwords_nltk(self) -> None:
        if not NLTK_AVAILABLE:
            return
        try:
            palavras = nltk_stopwords.words("portuguese")
            for p in palavras:
                norm = remover_acentos(p.lower())
                if norm not in STOPWORDS_PRESERVAR:
                    self._stopwords.add(norm)
        except LookupError:
            # Corpus stopwords do NLTK não baixado — mantém stopwords base + spaCy
            pass

        try:
            self._nltk_stemmer = RSLPStemmer()
        except LookupError:
            self._nltk_stemmer = None

    def _inicializar_spacy(self) -> Language:
        try:
            nlp = spacy.load(self.model_name)
            self.backend_nlp = f"spacy:{self.model_name}"
        except OSError:
            nlp = spacy.blank("pt")
            self.backend_nlp = (
                "spacy:blank_pt+nltk" if self._nltk_stemmer else "spacy:blank_pt"
            )

        # Garante que palavras críticas nunca sejam descartadas pelo is_stop do spaCy
        for palavra in STOPWORDS_PRESERVAR:
            nlp.vocab[palavra].is_stop = False
            if palavra == "nao":
                nlp.vocab["não"].is_stop = False
            elif palavra == "ate":
                nlp.vocab["até"].is_stop = False
        return nlp

    def _configurar_entity_ruler(self) -> None:
        """Adiciona EntityRuler ao pipeline spaCy para NER especializado de games e e-commerce."""
        if "entity_ruler" in self.nlp.pipe_names:
            ruler = self.nlp.get_pipe("entity_ruler")
        elif "ner" in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.add_pipe("entity_ruler")

        patterns: List[Dict[str, Any]] = []

        # Padrões de lojas digitais
        for alias in sorted(LOJAS_NER.keys(), key=len, reverse=True):
            tokens = alias.split()
            patterns.append(
                {
                    "label": "LOJA",
                    "pattern": [{"LOWER": tok} for tok in tokens],
                }
            )

        # Padrões de catálogo de jogos
        for titulo in sorted(CATALOGO_JOGOS_NER, key=len, reverse=True):
            doc_titulo = self.nlp.make_doc(titulo)
            patterns.append(
                {
                    "label": "JOGO",
                    "pattern": [{"LOWER": t.text.lower()} for t in doc_titulo],
                }
            )

        # Padrão para código de pedido corporativo (ex.: DH-1001)
        patterns.append(
            {
                "label": "CODIGO_PEDIDO",
                "pattern": [{"TEXT": {"REGEX": r"(?i)^DH-?\d{3,6}$"}}],
            }
        )

        # Padrão para CPF formatado ou numérico
        patterns.append(
            {
                "label": "CPF",
                "pattern": [
                    {"TEXT": {"REGEX": r"^\d{3}\.?\d{3}\.?\d{3}-?\d{2}$"}}
                ],
            }
        )

        ruler.add_patterns(patterns)

    # ------------------------------------------------------------------
    # Pré-processamento Estrutural (Normalização + Tokenização + Lematização)
    # ------------------------------------------------------------------

    def _normalizar_texto(self, texto: str) -> str:
        """
        Normaliza gírias, abreviações, apelidos de jogos e pequenos ruídos ortográficos
        mantendo intactos códigos de pedido, CPFs, e-mails e números.
        """
        ajustado = (texto or "").strip()
        for padrao, substituicao in APELIDOS_JOGOS:
            ajustado = padrao.sub(substituicao, ajustado)

        partes = re.findall(r"[\w@.\-:']+|\s+|[^\w\s]", ajustado, flags=re.UNICODE)
        resultado: List[str] = []

        for parte in partes:
            if not parte or parte.isspace() or not re.match(r"^[\w]+$", parte):
                resultado.append(parte)
                continue

            lower = parte.lower()
            sem_acento = remover_acentos(lower)

            # Preserva códigos (DH-1001), números ou nomes próprios capitalizados no meio da frase
            if lower.isdigit() or re.match(r"^dh\d+$", lower):
                resultado.append(parte)
                continue

            if lower in NORMALIZACAO_TOKENS:
                resultado.append(NORMALIZACAO_TOKENS[lower])
            elif sem_acento in NORMALIZACAO_TOKENS:
                resultado.append(NORMALIZACAO_TOKENS[sem_acento])
            elif (
                len(sem_acento) >= 5
                and not parte[0].isupper()
                and sem_acento not in self._stopwords
                and sem_acento not in PALAVRAS_COMUNS_NAO_CORRIGIR
                and not self.nlp.vocab[lower].is_stop
            ):
                proximas = difflib.get_close_matches(
                    sem_acento, VOCABULARIO_DOMINIO, n=1, cutoff=0.84
                )
                resultado.append(proximas[0] if proximas else parte)
            else:
                resultado.append(parte)

        return "".join(resultado)

    def _extrair_lemas(self, texto: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Executa o pipeline spaCy retornando (tokens, tokens_sem_stopwords, lemas).
        Se estiver em fallback sem modelo estatístico, aplica RSLPStemmer (NLTK) nos lemas.
        """
        doc = self.nlp(texto)
        tokens: List[str] = []
        sem_stopwords: List[str] = []
        lemas: List[str] = []

        usa_lema_spacy = self.backend_nlp.startswith("spacy:pt_")

        for tok in doc:
            if tok.is_space or tok.is_punct:
                continue
            bruto = tok.text.strip()
            if not bruto:
                continue
            tokens.append(bruto)

            # Preserva marcadores sintéticos de entidades (__jogo__, __valor__, etc.)
            if bruto.startswith("__") and bruto.endswith("__"):
                sem_stopwords.append(bruto.lower())
                lemas.append(bruto.lower())
                continue

            norm = remover_acentos(bruto.lower())
            eh_stop = (
                tok.is_stop or norm in self._stopwords
            ) and norm not in STOPWORDS_PRESERVAR
            if eh_stop:
                continue

            sem_stopwords.append(norm)

            if usa_lema_spacy and tok.lemma_:
                lema = remover_acentos(tok.lemma_.lower().strip())
            elif self._nltk_stemmer is not None and len(norm) >= 4:
                lema = remover_acentos(self._nltk_stemmer.stem(norm))
            else:
                lema = norm

            if lema:
                lemas.append(lema)

        return tokens, sem_stopwords, lemas

    # ------------------------------------------------------------------
    # Treinamento e Inferência do Classificador de Intenções (TF-IDF + Cosseno)
    # ------------------------------------------------------------------

    def _treinar_classificador(self) -> None:
        docs_por_intencao: List[Tuple[str, List[str]]] = []
        df: Counter[str] = Counter()

        for intencao, frases in EXEMPLOS_INTENCOES.items():
            for frase in frases:
                _, _, lemas = self._extrair_lemas(frase)
                if not lemas:
                    continue
                docs_por_intencao.append((intencao, lemas))
                for termo in set(lemas):
                    df[termo] += 1

        total_docs = max(len(docs_por_intencao), 1)
        self._idf = {
            termo: math.log((total_docs + 1) / (freq + 1)) + 1.0
            for termo, freq in df.items()
        }
        self._max_idf = math.log((total_docs + 1) / 1.0) + 1.0

        self._vetores_treinamento = []
        for intencao, lemas in docs_por_intencao:
            vetor, norma = self._vetorizar_tfidf(lemas)
            if norma > 0:
                self._vetores_treinamento.append((intencao, vetor, norma))

    def _vetorizar_tfidf(self, lemas: List[str]) -> Tuple[Dict[str, float], float]:
        """
        Constrói vetor TF-IDF atribuindo peso IDF máximo a termos fora do vocabulário,
        evitando falsos positivos em frases completamente fora do domínio.
        """
        contagem = Counter(lemas)
        total = max(len(lemas), 1)
        vetor: Dict[str, float] = {}
        soma_quadrados = 0.0

        for termo, freq in contagem.items():
            tf = freq / total
            idf = self._idf.get(termo, self._max_idf)
            peso = tf * idf
            vetor[termo] = peso
            soma_quadrados += peso * peso

        return vetor, math.sqrt(soma_quadrados)

    def _classificar_intencao(
        self, lemas_mascarados: List[str], entidades: Dict[str, Any]
    ) -> Tuple[str, float, List[Tuple[str, float]]]:
        if not lemas_mascarados:
            return "desconhecida", 0.0, []

        q_vec, q_norm = self._vetorizar_tfidf(lemas_mascarados)
        if q_norm == 0.0:
            return "desconhecida", 0.0, []

        melhores_por_intencao: Dict[str, float] = {}
        for intencao, t_vec, t_norm in self._vetores_treinamento:
            dot = sum(
                q_peso * t_vec[termo]
                for termo, q_peso in q_vec.items()
                if termo in t_vec
            )
            sim = dot / (q_norm * t_norm) if (q_norm * t_norm) > 0 else 0.0
            if sim > melhores_por_intencao.get(intencao, 0.0):
                melhores_por_intencao[intencao] = sim

        # Reforço contextual leve guiado pelas entidades NER extraídas
        if "codigo_pedido" in entidades and "solicitar_reembolso" not in (
            max(melhores_por_intencao, key=melhores_por_intencao.get, default=""),
        ):
            melhores_por_intencao["consultar_pedido"] = min(
                1.0, melhores_por_intencao.get("consultar_pedido", 0.0) + 0.18
            )
        if "opcao_numero" in entidades and len(lemas_mascarados) <= 5:
            melhores_por_intencao["selecionar_opcao"] = min(
                1.0, melhores_por_intencao.get("selecionar_opcao", 0.0) + 0.25
            )
        if "jogo" in entidades and "valor_brl" not in entidades:
            if melhores_por_intencao.get("criar_alerta", 0.0) < 0.45:
                melhores_por_intencao["consultar_preco"] = min(
                    1.0, melhores_por_intencao.get("consultar_preco", 0.0) + 0.12
                )

        ranking = sorted(
            melhores_por_intencao.items(), key=lambda x: x[1], reverse=True
        )
        melhor_intencao, melhor_score = ranking[0] if ranking else ("desconhecida", 0.0)

        if melhor_score < self.confidence_threshold:
            return "desconhecida", round(melhor_score, 3), ranking[:3]

        return melhor_intencao, round(melhor_score, 3), ranking[:3]

    # ------------------------------------------------------------------
    # Reconhecimento de Entidades Nomeadas (NER Híbrido)
    # ------------------------------------------------------------------

    def _extrair_entidades(
        self, texto_original: str, texto_normalizado: str
    ) -> Tuple[Dict[str, Any], str]:
        """
        Extrai entidades usando pipeline spaCy (EntityRuler + NER nativo) + regras
        contextuais e retorna (dicionario_entidades, texto_com_entidades_mascaradas).
        """
        entidades: Dict[str, Any] = {}
        texto_mascarado = texto_normalizado

        # 1) CPF
        m_cpf = re.search(r"\b(\d{3}\.?\d{3}\.?\d{3}-?\d{2})\b", texto_original)
        if m_cpf:
            entidades["cpf"] = m_cpf.group(1)
            texto_mascarado = re.sub(
                r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b", " __cpf__ ", texto_mascarado
            )

        # 2) E-mail
        m_email = re.search(
            r"\b([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\b", texto_original
        )
        if m_email:
            entidades["email"] = m_email.group(1)
            texto_mascarado = texto_mascarado.replace(m_email.group(1), " __email__ ")

        # 3) Código de Pedido (ex.: DH-1001, #1001, pedido 1002)
        m_ped = re.search(r"\b(DH-?\d{3,6})\b", texto_original, flags=re.IGNORECASE)
        if not m_ped:
            m_ped = re.search(
                r"(?:pedido|pedidu|#)\s*#?(\d{3,6})\b",
                texto_original,
                flags=re.IGNORECASE,
            )
        if m_ped:
            bruto_ped = m_ped.group(1).upper()
            if bruto_ped.isdigit():
                codigo = f"DH-{bruto_ped}"
            elif bruto_ped.startswith("DH") and "-" not in bruto_ped:
                codigo = f"DH-{bruto_ped[2:]}"
            else:
                codigo = bruto_ped
            entidades["codigo_pedido"] = codigo
            texto_mascarado = re.sub(
                re.escape(m_ped.group(1)), " __pedido__ ", texto_mascarado, flags=re.IGNORECASE
            )

        # 4) Valor monetário em BRL (ex.: R$ 50, 40 reais, ate 120,50)
        m_valor = re.search(
            r"(?:r\$\s*(\d+(?:[.,]\d{1,2})?))|"
            r"(?:(\d+(?:[.,]\d{1,2})?)\s*(?:reais|real|brl|conto|contos|pila|pilas|mangos))|"
            r"(?:(?:ate|até|abaixo\s+de|menos\s+de|por|para|pra)\s*(?:r\$\s*)?(\d+(?:[.,]\d{1,2})?)(?!\s*(?:horas?|dias?|vezes)))",
            texto_normalizado,
            flags=re.IGNORECASE,
        )
        if m_valor:
            numero_str = m_valor.group(1) or m_valor.group(2) or m_valor.group(3)
            if numero_str:
                try:
                    entidades["valor_brl"] = float(numero_str.replace(",", "."))
                    texto_mascarado = texto_mascarado.replace(
                        m_valor.group(0), " __valor__ "
                    )
                except ValueError:
                    pass

        # 5) Número de opção em fluxo Stateful (ex.: "opção 2", "o primeiro", "link do 3")
        ordinais = {
            "primeiro": 1,
            "primeira": 1,
            "segundo": 2,
            "segunda": 2,
            "terceiro": 3,
            "terceira": 3,
            "quarto": 4,
            "quarta": 4,
            "quinto": 5,
            "quinta": 5,
        }
        m_opc = re.search(
            r"(?:opcao|opção|item|oferta|numero|link\s+(?:da|do)|detalhes?\s+(?:da|do)|escolho\s+(?:a|o)?)\s*(?:numero\s*)?([1-5])\b",
            remover_acentos(texto_normalizado.lower()),
        )
        if m_opc:
            entidades["opcao_numero"] = int(m_opc.group(1))
            texto_mascarado = re.sub(
                r"\b" + m_opc.group(1) + r"\b", " __opcao__ ", texto_mascarado, count=1
            )
        else:
            for palavra_ord, num_ord in ordinais.items():
                if re.search(
                    rf"\b{palavra_ord}\b", remover_acentos(texto_normalizado.lower())
                ):
                    entidades["opcao_numero"] = num_ord
                    texto_mascarado = re.sub(
                        rf"\b{palavra_ord}\b",
                        " __opcao__ ",
                        texto_mascarado,
                        flags=re.IGNORECASE,
                    )
                    break

        # 6) Pipeline spaCy (EntityRuler + NER nativo) para LOJA e JOGO
        doc = self.nlp(texto_normalizado)
        for ent in doc.ents:
            if ent.label_ == "LOJA" and "loja" not in entidades:
                chave_loja = ent.text.strip().lower()
                entidades["loja"] = LOJAS_NER.get(chave_loja, ent.text.strip())
                texto_mascarado = re.sub(
                    re.escape(ent.text),
                    " __loja__ ",
                    texto_mascarado,
                    flags=re.IGNORECASE,
                )
            elif ent.label_ == "JOGO" and "jogo" not in entidades:
                entidades["jogo"] = ent.text.strip()
                texto_mascarado = re.sub(
                    re.escape(ent.text),
                    " __jogo__ ",
                    texto_mascarado,
                    flags=re.IGNORECASE,
                )

        # 7) Extração dinâmica de JOGO não listado no catálogo (para qualquer título na CheapShark!)
        if "jogo" not in entidades:
            candidato = self._extrair_jogo_dinamico(texto_mascarado, doc)
            if candidato:
                entidades["jogo"] = candidato
                texto_mascarado = re.sub(
                    re.escape(candidato),
                    " __jogo__ ",
                    texto_mascarado,
                    flags=re.IGNORECASE,
                )

        return entidades, texto_mascarado

    def _extrair_jogo_dinamico(self, texto_mascarado: str, doc: Any) -> Optional[str]:
        """
        Captura títulos de jogos que não estão no catálogo fixo do EntityRuler
        usando padrões sintáticos após gatilhos de consulta e entidades nativas do spaCy.
        """
        padroes = [
            re.compile(
                r"\b(?:preco|valor|custar|custa|custando|alerta|avisar|notificar|buscar|pesquisar|procurar|comprar|sobre|tem)\s+"
                r"(?:de\s+preco\s+)?(?:(?:de|do|da|pro|pra|para|o|a)\s+)?"
                r"(?P<jogo>[A-Za-z0-9][A-Za-z0-9\s:'\-&]{1,42})",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bquanto\s+(?:esta|ta|sai|sair|fica)\s+(?:(?:o|a)\s+)?"
                r"(?P<jogo>[A-Za-z0-9][A-Za-z0-9\s:'\-&]{1,42})",
                flags=re.IGNORECASE,
            ),
        ]

        for padrao in padroes:
            m = padrao.search(texto_mascarado)
            if not m:
                continue
            candidato = self._limpar_candidato_jogo(m.group("jogo"))
            if candidato:
                return candidato

        # Fallback via NER nativo do spaCy (MISC / WORK_OF_ART) se o modelo estatístico estiver ativo
        for ent in doc.ents:
            if ent.label_ in {"MISC", "WORK_OF_ART"}:
                candidato = self._limpar_candidato_jogo(ent.text)
                if candidato and candidato.lower() not in LOJAS_NER:
                    return candidato

        return None

    @staticmethod
    def _limpar_candidato_jogo(bruto: str) -> Optional[str]:
        # Remove marcadores internos e caudas de filtro (ex.: "na __loja__", "ate __valor__")
        texto = re.split(
            r"__\w+__|\b(?:na|no|em|ate|até|por|para|pra|hoje|agora|baixar|cair|com|que|quando|favor|em promocao)\b",
            bruto,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        texto = texto.strip(" ?!.,;:-")
        if not texto or len(texto) < 2:
            return None

        palavras = [remover_acentos(p.lower()) for p in re.findall(r"[A-Za-zÀ-ÿ0-9]+", texto)]
        if not palavras:
            return None

        # Rejeita se todas as palavras forem genéricas ou não-jogo
        if all(p in TERMOS_NAO_JOGO or p in LOJAS_NER for p in palavras):
            return None

        # Rejeita se parecer pedido/CPF numérico solto
        if len(palavras) == 1 and palavras[0].isdigit():
            return None

        return texto

    # ------------------------------------------------------------------
    # Análise de Sentimento e Urgência (Eixo 4 — Transbordo Humano)
    # ------------------------------------------------------------------

    def _analisar_sentimento(
        self, texto_original: str, texto_normalizado: str, lemas: List[str]
    ) -> Tuple[str, float]:
        """
        Calcula polaridade e nível de insatisfação/urgência.
        Retorna (sentimento, score_insatisfacao entre 0.0 e 1.0).
        """
        norm_sem_acento = remover_acentos(texto_normalizado.lower())
        score = 0.0

        # Checa expressões compostas críticas
        for frase, peso in FRASES_CRITICAS_URGENCIA:
            if frase in norm_sem_acento:
                score += peso

        # Checa termos individuais críticos (tanto nos tokens quanto nos lemas)
        tokens_simples = set(re.findall(r"[a-z]+", norm_sem_acento)) | set(lemas)
        for termo in tokens_simples:
            if termo in TERMOS_CRITICOS_URGENCIA:
                score += TERMOS_CRITICOS_URGENCIA[termo]

        # Ênfase emocional: excesso de exclamações ou frase inteira em CAIXA ALTA
        if texto_original.count("!") >= 2:
            score += 0.15
        letras = [c for c in texto_original if c.isalpha()]
        if len(letras) >= 12 and sum(1 for c in letras if c.isupper()) / len(letras) > 0.75:
            score += 0.20

        score = round(min(score, 1.0), 2)

        if score >= config.URGENCY_SCORE_THRESHOLD:
            return "CRITICO_URGENTE", score
        if score >= 0.30:
            return "NEGATIVO", score

        if tokens_simples & TERMOS_POSITIVOS:
            return "POSITIVO", 0.0

        return "NEUTRO", score

    # ------------------------------------------------------------------
    # Ponto de Entrada Público
    # ------------------------------------------------------------------

    def analisar(self, mensagem: str) -> NLUResult:
        """Executa todo o pipeline NLU sobre a mensagem do usuário."""
        texto_original = (mensagem or "").strip()
        texto_normalizado = self._normalizar_texto(texto_original)

        tokens, tokens_sem_sw, lemas = self._extrair_lemas(texto_normalizado)
        entidades, texto_mascarado = self._extrair_entidades(
            texto_original, texto_normalizado
        )
        _, _, lemas_mascarados = self._extrair_lemas(texto_mascarado)

        intencao, confianca, ranking = self._classificar_intencao(
            lemas_mascarados, entidades
        )
        sentimento, score_insatisfacao = self._analisar_sentimento(
            texto_original, texto_normalizado, lemas
        )

        return NLUResult(
            texto_original=texto_original,
            texto_normalizado=texto_normalizado,
            tokens=tokens,
            tokens_sem_stopwords=tokens_sem_sw,
            lemas=lemas,
            intencao=intencao,
            confianca=confianca,
            entidades=entidades,
            sentimento=sentimento,
            score_insatisfacao=score_insatisfacao,
            backend_nlp=self.backend_nlp,
            ranking_intencoes=ranking,
        )
