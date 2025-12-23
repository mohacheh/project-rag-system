"""
llm_chain.py – LLM-Kommunikation, Prompt Engineering und Antwortgenerierung.

WARUM PROMPT ENGINEERING WICHTIG IST:
Das LLM "weiß" nichts über unser Dokument aus eigener Erfahrung.
Es muss explizit angewiesen werden:
  1. NUR den gegebenen Kontext zu nutzen (keine Halluzinationen)
  2. Quellenangaben zu machen
  3. Unsicherheiten klar zu kommunizieren

Ein schlechter Prompt → das Modell erfindet Informationen.
Ein guter Prompt → das Modell gibt präzise, belegbare Antworten.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI, APIError, RateLimitError, AuthenticationError

import config
from retriever import RetrievalResult

logger = logging.getLogger(__name__)


# Striktes System-Prompt das Halluzinationen verhindert.
# "Streng quellenbasiert" ist entscheidend für RAG-Anwendungen:
# Ohne diese Anweisung neigen LLMs dazu, ihr Vorwissen zu mischen,
# was zu nicht verifizierbaren Aussagen führt.
SYSTEM_PROMPT = """Du bist ein präziser Dokumenten-Assistent.

REGELN (nicht verhandelbar):
1. Beantworte Fragen NUR auf Basis der bereitgestellten Kontext-Abschnitte.
2. Wenn die Antwort nicht im Kontext steht, sage: "Diese Information ist im Dokument nicht enthalten."
3. Gib am Ende deiner Antwort immer die verwendeten Abschnitte an (z.B. "[Abschnitt 1, 3]").
4. Antworte auf Deutsch, präzise und strukturiert.
5. Erfinde KEINE Informationen, auch wenn sie plausibel klingen.

FORMAT:
- Antwort: [Deine Antwort basierend auf dem Kontext]
- Verwendet: [Abschnittsnummern]
"""


@dataclass
class LLMResponse:
    """
    Strukturierte Antwort des LLMs mit Metadaten.

    Token-Tracking ist für Kosten-Bewusstsein wichtig:
    GPT-3.5-turbo kostet ~$0.002 pro 1000 Tokens.
    Bei intensiver Nutzung können schnell Kosten entstehen.
    """

    answer: str
    sources: list
    # Token-Verbrauch für Kosten-Transparenz
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost_usd(self) -> float:
        """
        Schätzt die Kosten dieser Anfrage in USD.

        GPT-3.5-turbo Preise (Stand 2024, kann sich ändern):
        - Input: $0.0005 / 1K Tokens
        - Output: $0.0015 / 1K Tokens
        """
        input_cost = (self.prompt_tokens / 1000) * 0.0005
        output_cost = (self.completion_tokens / 1000) * 0.0015
        return input_cost + output_cost


class LLMChain:
    """
    Verwaltet die Kommunikation mit dem OpenAI LLM.

    Verwendet die neue openai Python-Bibliothek (v1.0+) direkt,
    da sie stabiler und übersichtlicher als die LangChain-Abstraktion ist
    für einfache Chat-Completions.
    """

    def __init__(
        self,
        model: str = config.LLM_MODEL,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialisiert den LLM-Client.

        Args:
            model: OpenAI-Modell-ID.
            temperature: Kreativität (0=deterministisch, 1=kreativ).
            max_tokens: Maximale Länge der generierten Antwort.
            api_key: OpenAI API-Key (Standard: aus config.py / .env).
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # OpenAI-Client initialisieren – wirft AuthenticationError wenn Key fehlt
        self.client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)

        # Gesamte Token-Statistik über die Session
        self.total_tokens_used = 0
        self.total_cost_usd = 0.0

        logger.info(f"🤖 LLM-Chain initialisiert: {model} (temp={temperature})")

    def generate_answer(self, retrieval_result: RetrievalResult) -> LLMResponse:
        """
        Generiert eine Antwort basierend auf dem Retrieval-Ergebnis.

        Die Prompt-Struktur folgt dem RAG-Standard:
        System-Prompt → Kontext → Nutzerfrage

        Warum den Kontext im User-Turn statt im System-Prompt?
        Es hat sich empirisch gezeigt, dass Modelle Kontext im User-Turn
        besser "beachten" und seltener ignorieren.

        Args:
            retrieval_result: Ergebnis der Vektordatenbank-Suche.

        Returns:
            LLMResponse mit Antwort und Token-Statistiken.

        Raises:
            RuntimeError: Bei API-Fehlern (Rate Limit, Auth, etc.).
        """
        if not retrieval_result.found_results:
            return LLMResponse(
                answer=(
                    "Ich konnte keine relevanten Informationen zu deiner Frage "
                    "in dem Dokument finden. Bitte formuliere die Frage anders "
                    "oder überprüfe, ob das Thema im Dokument behandelt wird."
                ),
                sources=[],
            )

        # User-Nachricht aufbauen: Kontext + Frage klar trennen
        user_message = self._build_user_message(retrieval_result)

        logger.debug(f"Sende Anfrage an {self.model}...")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except AuthenticationError:
            raise RuntimeError(
                "OpenAI Authentication fehlgeschlagen!\n"
                "Prüfe deinen API-Key in der .env-Datei."
            )
        except RateLimitError:
            raise RuntimeError(
                "OpenAI Rate-Limit erreicht!\n"
                "Bitte warte kurz und versuche es erneut, "
                "oder upgraden deinen OpenAI-Plan."
            )
        except APIError as e:
            raise RuntimeError(f"OpenAI API-Fehler: {e}")

        # Antwort und Token-Statistiken extrahieren
        answer = response.choices[0].message.content or ""
        usage = response.usage

        llm_response = LLMResponse(
            answer=answer,
            sources=retrieval_result.sources,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

        # Session-Statistiken aktualisieren
        self.total_tokens_used += llm_response.total_tokens
        self.total_cost_usd += llm_response.estimated_cost_usd

        logger.debug(
            f"  Token-Verbrauch: {llm_response.total_tokens} "
            f"(≈ ${llm_response.estimated_cost_usd:.4f})"
        )

        return llm_response

    def _build_user_message(self, retrieval_result: RetrievalResult) -> str:
        """
        Baut die formatierte User-Nachricht mit Kontext und Frage.

        Klare Trennung zwischen Kontext und Frage hilft dem Modell,
        die Aufgabe besser zu verstehen und weniger zu "halluzinieren".

        Args:
            retrieval_result: RetrievalResult mit Kontext und Frage.

        Returns:
            Formatierte User-Nachricht.
        """
        return (
            f"KONTEXT AUS DEM DOKUMENT:\n"
            f"{'=' * 50}\n"
            f"{retrieval_result.context}\n"
            f"{'=' * 50}\n\n"
            f"FRAGE: {retrieval_result.query}"
        )

    def get_session_stats(self) -> dict:
        """
        Gibt Token- und Kosten-Statistiken der aktuellen Session zurück.

        Returns:
            Dict mit Token-Anzahl und geschätzten Kosten.
        """
        return {
            "total_tokens": self.total_tokens_used,
            "estimated_cost_usd": self.total_cost_usd,
            "model": self.model,
        }
