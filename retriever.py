"""
retriever.py – Suchanfragen, Ranking und Kontext-Aufbereitung.

Der Retriever ist die Brücke zwischen Nutzerfrage und Vektordatenbank.
Er übernimmt:
  1. Suche relevanter Chunks
  2. Deduplizierung ähnlicher Chunks (verhindert Redundanz im Kontext)
  3. Formatierung des Kontexts für das LLM
  4. Extraktion der Quellenangaben für die Antwort
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from langchain.schema import Document

import config
from vector_store import VectorStore
from utils import format_similarity_score

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """
    Strukturiertes Ergebnis einer Suchanfrage.

    Ein Dataclass statt Dict macht den Code robuster:
    Tippfehler in Schlüsselnamen führen zu AttributeErrors statt stillschweigend
    None zurückzugeben.
    """

    query: str
    documents: List[Document] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    context: str = ""
    sources: List[str] = field(default_factory=list)

    @property
    def found_results(self) -> bool:
        """True wenn mindestens ein relevanter Chunk gefunden wurde."""
        return len(self.documents) > 0

    @property
    def top_score(self) -> float:
        """Bester Similarity-Score (0 wenn keine Ergebnisse)."""
        return max(self.scores) if self.scores else 0.0


class Retriever:
    """
    Sucht relevante Dokumenten-Chunks für eine Nutzerfrage.

    Die Retriever-Klasse abstrahiert die Suchlogik vom Rest der Anwendung.
    Das ermöglicht später einfachen Austausch gegen erweiterte Strategien
    (z.B. Hybrid-Search mit Volltextsuche + Vektorsuche).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = config.TOP_K_RESULTS,
        min_similarity: float = config.MIN_SIMILARITY,
    ) -> None:
        """
        Initialisiert den Retriever.

        Args:
            vector_store: Instanz des VectorStore für Datenbankzugriffe.
            top_k: Anzahl der zu holenden Chunks pro Anfrage.
            min_similarity: Mindest-Ähnlichkeit für Chunk-Aufnahme.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Führt die vollständige Retrieval-Pipeline für eine Frage aus.

        Pipeline:
        1. Semantische Suche in der Vektordatenbank
        2. Deduplizierung sehr ähnlicher Chunks
        3. Kontext-String für LLM zusammenbauen
        4. Quellenangaben extrahieren

        Args:
            query: Natürlichsprachige Frage des Nutzers.

        Returns:
            RetrievalResult mit Chunks, Scores, Kontext und Quellen.
        """
        logger.info(f"🔍 Suche relevante Abschnitte für: '{query[:60]}...'")

        result = RetrievalResult(query=query)

        try:
            # Semantische Ähnlichkeitssuche in ChromaDB
            raw_results: List[Tuple[Document, float]] = (
                self.vector_store.similarity_search(
                    query=query,
                    k=self.top_k,
                    min_similarity=self.min_similarity,
                )
            )
        except RuntimeError as e:
            logger.error(str(e))
            return result

        if not raw_results:
            logger.warning("  ⚠️  Keine relevanten Chunks gefunden")
            return result

        # Deduplizierung: Sehr ähnliche Chunks (>90% Textübereinstimmung) entfernen.
        # Bei Overlap-Chunking kann es vorkommen, dass fast identische Chunks
        # zurückgegeben werden – die würden den LLM-Kontext aufblähen ohne Mehrwert.
        deduplicated = self._deduplicate(raw_results, similarity_threshold=0.90)

        # Ergebnisse im RetrievalResult speichern
        for doc, score in deduplicated:
            result.documents.append(doc)
            result.scores.append(score)

        # Kontext für das LLM formatieren (nummerierte Abschnitte)
        result.context = self._build_context(deduplicated)

        # Quellenangaben für die Antwort extrahieren (z.B. "Seite 3, Seite 7")
        result.sources = self._extract_sources(deduplicated)

        # Suchergebnis-Zusammenfassung loggen
        scores_str = ", ".join(
            format_similarity_score(s) for s in result.scores
        )
        logger.info(
            f"  📚 {len(result.documents)} relevante Chunks gefunden "
            f"(Similarity: {scores_str})"
        )

        return result

    def _deduplicate(
        self,
        results: List[Tuple[Document, float]],
        similarity_threshold: float = 0.90,
    ) -> List[Tuple[Document, float]]:
        """
        Entfernt nahezu identische Chunks aus den Suchergebnissen.

        Methode: Zeichenübereinstimmungs-Ratio zwischen Chunk-Texten.
        Jaccard-Similarity ist einfacher zu berechnen als komplexere Metriken
        und für unseren Zweck (grobe Duplikat-Erkennung) ausreichend.

        Args:
            results: Liste von (Document, score) Tupeln.
            similarity_threshold: Ab welcher Textübereinstimmung als Duplikat gilt.

        Returns:
            Deduplizierte Liste.
        """
        if len(results) <= 1:
            return results

        unique_results = [results[0]]  # Erstes Ergebnis (höchster Score) behalten

        for doc, score in results[1:]:
            is_duplicate = False
            for existing_doc, _ in unique_results:
                text_similarity = self._jaccard_similarity(
                    doc.page_content, existing_doc.page_content
                )
                if text_similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(
                        f"  Duplikat entfernt (Text-Similarity: {text_similarity:.2f})"
                    )
                    break

            if not is_duplicate:
                unique_results.append((doc, score))

        return unique_results

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Berechnet die Jaccard-Ähnlichkeit zweier Texte basierend auf Wortmengen.

        Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        Einfach, schnell und für Duplikat-Erkennung ausreichend genau.

        Args:
            text1: Erster Text.
            text2: Zweiter Text.

        Returns:
            Ähnlichkeit zwischen 0.0 (komplett unterschiedlich) und 1.0 (identisch).
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _build_context(self, results: List[Tuple[Document, float]]) -> str:
        """
        Baut den Kontext-String für das LLM aus den gefundenen Chunks.

        Struktur: Nummerierte Abschnitte mit Quelle und Inhalt.
        Diese klare Struktur hilft dem LLM, den Kontext zu verstehen
        und korrekte Quellenangaben zu machen.

        Args:
            results: Liste von (Document, score) Tupeln.

        Returns:
            Formatierter Kontext-String.
        """
        context_parts = []

        for i, (doc, score) in enumerate(results, start=1):
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "Unbekannt")

            section = (
                f"[Abschnitt {i}] (Quelle: {source}, Seite {page})\n"
                f"{doc.page_content}"
            )
            context_parts.append(section)

        return "\n\n---\n\n".join(context_parts)

    def _extract_sources(self, results: List[Tuple[Document, float]]) -> List[str]:
        """
        Extrahiert eindeutige Quellenangaben aus den Suchergebnissen.

        Args:
            results: Liste von (Document, score) Tupeln.

        Returns:
            Sortierte Liste von Quellenangaben (z.B. ["Seite 3", "Seite 7"]).
        """
        sources = []
        seen = set()

        for doc, _ in results:
            page = doc.metadata.get("page", "?")
            source = doc.metadata.get("source", "")
            key = f"{source}_p{page}"

            if key not in seen:
                seen.add(key)
                sources.append(f"Seite {page}")

        return sorted(sources, key=lambda s: int(s.split()[-1]) if s.split()[-1].isdigit() else 0)
