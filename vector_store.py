"""
vector_store.py – Embedding-Berechnung und ChromaDB-Vektordatenbank.

WIE FUNKTIONIEREN VEKTOREN IN RAG?
Jeder Text-Chunk wird in einen hochdimensionalen Zahlenvektor umgewandelt
(z.B. 384 Dimensionen mit MiniLM). Ähnliche Texte haben ähnliche Vektoren.
Beim Suchen wird die Frage ebenfalls in einen Vektor umgewandelt und die
nächsten Nachbarn im Vektorraum gefunden – das ist semantische Suche.

Beispiel: "Auto" und "Fahrzeug" sind nahe beieinander im Vektorraum,
auch wenn keine Wörter übereinstimmen (wie bei klassischer Volltextsuche).
"""

import hashlib
import logging
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Verwaltet die Vektordatenbank: Embeddings erstellen, speichern, suchen.

    ChromaDB wurde gewählt weil:
    - Persistent: Datenbank bleibt nach Programmende erhalten (kein erneutes Embedding)
    - Einfach: Läuft lokal, keine externe Infrastruktur nötig
    - LangChain-Integration: Native Unterstützung
    - Skalierbar bis zu Millionen von Vektoren für ein Portfolio-Projekt mehr als genug
    """

    def __init__(
        self,
        persist_dir: str = config.CHROMA_PERSIST_DIR,
        collection_name: str = config.COLLECTION_NAME,
        embedding_model: str = config.EMBEDDING_MODEL,
    ) -> None:
        """
        Initialisiert den VectorStore und lädt das Embedding-Modell.

        Das Embedding-Modell wird beim ersten Aufruf von HuggingFace heruntergeladen
        (~90MB für MiniLM) und danach lokal gecacht.

        Args:
            persist_dir: Verzeichnis für persistente ChromaDB-Daten.
            collection_name: Name der Kollektion in ChromaDB.
            embedding_model: HuggingFace Model-ID für Embeddings.
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        logger.info(f"🔢 Lade Embedding-Modell: {embedding_model}")
        logger.info("  (Beim ersten Start: Download ~90MB, dann gecacht)")

        # HuggingFaceEmbeddings lädt das Modell lokal – kein API-Call, keine Kosten.
        # model_kwargs={"device": "cpu"} erzwingt CPU-Nutzung für Kompatibilität.
        # Wer eine GPU hat: "cuda" einsetzen für ~10x schnellere Embedding-Berechnung.
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "cpu"},
            # normalize_embeddings=True verbessert die Cosine-Similarity-Berechnung
            encode_kwargs={"normalize_embeddings": True},
        )

        # ChromaDB-Instanz initialisieren (oder bestehende laden)
        self.db: Optional[Chroma] = self._load_or_create_db()
        logger.info("  ✅ VectorStore bereit")

    def _load_or_create_db(self) -> Chroma:
        """
        Lädt eine bestehende ChromaDB oder erstellt eine neue.

        Returns:
            ChromaDB-Instanz.
        """
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )

    def add_documents(
        self, documents: List[Document], batch_size: int = 32
    ) -> int:
        """
        Fügt Dokumente zur Vektordatenbank hinzu, überspringt Duplikate.

        PERFORMANCE-TIPP: Batch-Verarbeitung ist entscheidend!
        Einzelne Embeddings berechnen wäre ~10-50x langsamer als Batches,
        da das Modell für jeden Batch nur einmal geladen werden muss.

        Duplikat-Erkennung via Content-Hash verhindert, dass dieselbe PDF
        mehrfach indiziert wird (z.B. wenn das Programm zweimal mit der
        gleichen Datei aufgerufen wird).

        Args:
            documents: Liste von LangChain-Documents.
            batch_size: Anzahl der Dokumente pro Embedding-Batch.
                       32 ist ein guter Kompromiss zwischen Geschwindigkeit
                       und Speicherverbrauch.

        Returns:
            Anzahl der neu hinzugefügten Dokumente.
        """
        # Bestehende Document-IDs abrufen, um Duplikate zu verhindern
        existing_ids = self._get_existing_ids()
        logger.debug(f"Bestehende Dokumente in DB: {len(existing_ids)}")

        # Nur neue Dokumente hinzufügen (Content-Hash als eindeutige ID)
        new_docs = []
        new_ids = []
        skipped = 0

        for doc in documents:
            doc_id = self._compute_doc_id(doc)
            if doc_id in existing_ids:
                skipped += 1
                continue
            new_docs.append(doc)
            new_ids.append(doc_id)

        if skipped > 0:
            logger.info(f"  ⏭️  {skipped} bereits vorhandene Chunks übersprungen")

        if not new_docs:
            logger.info("  ℹ️  Keine neuen Dokumente zum Indexieren")
            return 0

        logger.info(f"  🔢 Berechne Embeddings für {len(new_docs)} Chunks...")

        # Batch-Verarbeitung mit Fortschrittsbalken
        # tqdm zeigt einen visuellen Fortschrittsbalken in der Konsole
        added_count = 0
        for i in tqdm(
            range(0, len(new_docs), batch_size),
            desc="  Embeddings",
            unit="batch",
            ncols=60,
        ):
            batch_docs = new_docs[i : i + batch_size]
            batch_ids = new_ids[i : i + batch_size]

            self.db.add_documents(documents=batch_docs, ids=batch_ids)
            added_count += len(batch_docs)

        logger.info(f"  ✅ {added_count} neue Chunks zur Vektordatenbank hinzugefügt")
        return added_count

    def similarity_search(
        self,
        query: str,
        k: int = config.TOP_K_RESULTS,
        min_similarity: float = config.MIN_SIMILARITY,
    ) -> List[Tuple[Document, float]]:
        """
        Sucht die k ähnlichsten Chunks zur Suchanfrage.

        Die Suche funktioniert so:
        1. Frage → Embedding-Vektor (gleicher Raum wie Dokument-Vektoren)
        2. Cosine-Similarity zwischen Frage-Vektor und allen Dokument-Vektoren
        3. Top-K mit höchster Ähnlichkeit zurückgeben

        Args:
            query: Suchanfrage des Nutzers.
            k: Anzahl der zurückzugebenden Chunks.
            min_similarity: Mindest-Ähnlichkeit (0-1). Chunks darunter werden verworfen.

        Returns:
            Liste von (Document, similarity_score) Tupeln, absteigend nach Score.

        Raises:
            RuntimeError: Wenn die Datenbank leer ist.
        """
        if self.is_empty():
            raise RuntimeError(
                "Die Vektordatenbank ist leer. "
                "Bitte zuerst ein PDF mit --pdf indexieren."
            )

        # similarity_search_with_relevance_scores gibt (Document, score) Paare zurück
        # Score 1.0 = perfekte Übereinstimmung, 0.0 = keine Ähnlichkeit
        results = self.db.similarity_search_with_relevance_scores(query=query, k=k)

        # Ergebnisse unter dem Schwellwert herausfiltern (Rauschen reduzieren)
        filtered = [
            (doc, score) for doc, score in results if score >= min_similarity
        ]

        if not filtered:
            logger.warning(
                f"Keine Chunks mit Similarity >= {min_similarity} gefunden. "
                f"Versuche eine andere Frageformulierung oder reduziere MIN_SIMILARITY."
            )

        return filtered

    def is_empty(self) -> bool:
        """
        Prüft, ob die Vektordatenbank Dokumente enthält.

        Returns:
            True wenn leer, False wenn Dokumente vorhanden.
        """
        try:
            count = self.db._collection.count()
            return count == 0
        except Exception:
            return True

    def get_document_count(self) -> int:
        """
        Gibt die Anzahl der gespeicherten Chunks zurück.

        Returns:
            Anzahl der Chunks in der Datenbank.
        """
        try:
            return self.db._collection.count()
        except Exception:
            return 0

    def _get_existing_ids(self) -> set:
        """
        Lädt alle vorhandenen Document-IDs aus ChromaDB.

        Returns:
            Set von vorhandenen IDs.
        """
        try:
            result = self.db._collection.get(include=[])
            return set(result.get("ids", []))
        except Exception:
            return set()

    def _compute_doc_id(self, doc: Document) -> str:
        """
        Berechnet eine eindeutige ID für ein Dokument basierend auf seinem Inhalt.

        Warum Content-Hashing statt zufälliger IDs?
        Bei identischem Inhalt ergibt sich dieselbe ID → automatische Deduplizierung
        ohne eine separate Tracking-Datenbank zu pflegen.

        Args:
            doc: LangChain-Document-Objekt.

        Returns:
            MD5-Hash des Inhalts + Quelldatei als eindeutige ID.
        """
        content = f"{doc.metadata.get('source', '')}__{doc.page_content}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()
