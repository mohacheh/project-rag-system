"""
pdf_processor.py – PDF-Verarbeitung und intelligentes Text-Chunking.

WARUM CHUNKING?
LLMs haben ein begrenztes "Kontextfenster" (max. Tokens pro Anfrage).
Wir können kein 100-seitiges PDF direkt übergeben. Die Lösung:
  1. PDF → einzelne Text-Chunks zerlegen
  2. Chunks in Vektordatenbank speichern
  3. Bei Anfrage: nur die RELEVANTESTEN Chunks ans LLM schicken

Das ist das Herzstück von RAG: retrieval before generation.
"""

import logging
from pathlib import Path
from typing import List

import fitz  # PyMuPDF – schneller und zuverlässiger als pdfminer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import config

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Lädt PDF-Dateien und zerlegt sie in semantisch sinnvolle Chunks.

    Die Klasse kapselt die gesamte Dokumenten-Vorverarbeitungs-Pipeline:
    PDF lesen → Text extrahieren → Chunken → Metadaten anreichern.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> None:
        """
        Initialisiert den PDF-Prozessor mit Chunking-Parametern.

        Args:
            chunk_size: Maximale Zeichen pro Chunk (nicht Token! LangChain
                       rechnet in Zeichen, ca. 4 Zeichen ≈ 1 Token).
            chunk_overlap: Überlappung zwischen benachbarten Chunks in Zeichen.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter versucht zuerst bei Absätzen zu trennen,
        # dann bei Sätzen, dann bei Wörtern – NIE mitten in einem Wort.
        # Das erhält den semantischen Zusammenhang besser als naive Aufteilung.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Trennhierarchie: Absatz > Satz > Wort > Zeichen
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            # Trenner in den Chunk einschließen erhält den vollen Satz
            keep_separator=True,
        )

        logger.debug(
            f"PDFProcessor initialisiert: chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap}"
        )

    def load_and_split(self, pdf_path: str | Path) -> List[Document]:
        """
        Liest eine PDF-Datei und gibt eine Liste von LangChain-Documents zurück.

        Jedes Document enthält:
        - page_content: Den Textinhalt des Chunks
        - metadata: Seite, Quelldatei, Chunk-Index (für Quellen-Angaben)

        Args:
            pdf_path: Pfad zur PDF-Datei.

        Returns:
            Liste von Document-Objekten, bereit für das Embedding.

        Raises:
            FileNotFoundError: Wenn die PDF-Datei nicht gefunden wird.
            RuntimeError: Bei Fehler beim Lesen der PDF.
        """
        pdf_path = Path(pdf_path)
        logger.info(f"📄 Lade PDF: {pdf_path.name}")

        # Schritt 1: Rohtext + Metadaten seitenweise extrahieren
        raw_pages = self._extract_pages(pdf_path)

        if not raw_pages:
            raise RuntimeError(
                f"Keine Textinhalte in '{pdf_path.name}' gefunden. "
                "Möglicherweise ist die PDF gescannt (enthält nur Bilder). "
                "Für gescannte PDFs wäre OCR nötig (z.B. Tesseract)."
            )

        total_chars = sum(len(p["text"]) for p in raw_pages)
        logger.info(
            f"  ✅ {len(raw_pages)} Seiten gelesen, "
            f"{total_chars:,} Zeichen extrahiert"
        )

        # Schritt 2: Text chunken und Metadaten an jeden Chunk weitergeben
        documents = self._create_chunks(raw_pages, pdf_path.name)

        logger.info(f"  ✅ {len(documents)} Chunks erstellt")
        return documents

    def _extract_pages(self, pdf_path: Path) -> List[dict]:
        """
        Extrahiert den Rohtext jeder Seite mit PyMuPDF (fitz).

        PyMuPDF ist schneller als pdfminer und unterstützt mehr PDF-Formate.
        Es gibt den Text seitenweise zurück, was uns ermöglicht,
        die Seitenzahl als Metadatum zu speichern.

        Args:
            pdf_path: Pfad zur PDF-Datei.

        Returns:
            Liste von Dicts mit 'text' und 'page_number'.

        Raises:
            FileNotFoundError: Wenn die Datei nicht existiert.
            RuntimeError: Bei korrupter oder passwortgeschützter PDF.
        """
        pages = []

        try:
            doc = fitz.open(str(pdf_path))
        except fitz.FileNotFoundError:
            raise FileNotFoundError(f"PDF-Datei nicht gefunden: {pdf_path}")
        except Exception as e:
            raise RuntimeError(
                f"Fehler beim Öffnen der PDF: {e}\n"
                "Mögliche Ursachen: Korrupte Datei oder Passwortschutz."
            )

        with doc:  # Kontextmanager stellt sicher, dass die Datei geschlossen wird
            for page_num in range(len(doc)):
                page = doc[page_num]

                # get_text("text") extrahiert Klartext ohne Layout-Formatierung
                # Alternative: "blocks" für strukturiertere Extraktion
                text = page.get_text("text")

                # Leere oder sehr kurze Seiten überspringen (z.B. Titelseiten mit Bild)
                if len(text.strip()) < 50:
                    logger.debug(f"  Seite {page_num + 1} übersprungen (zu wenig Text)")
                    continue

                # Text normalisieren: mehrfache Leerzeilen auf eine reduzieren
                text = self._clean_text(text)

                pages.append({
                    "text": text,
                    "page_number": page_num + 1,  # 1-basiert für menschliche Lesbarkeit
                })

        return pages

    def _clean_text(self, text: str) -> str:
        """
        Bereinigt extrahierten PDF-Text von häufigen Artefakten.

        PDF-Extraktion erzeugt oft:
        - Überflüssige Leerzeilen (3+ aufeinander)
        - Führende/nachfolgende Leerzeichen pro Zeile
        - Sonderzeichen aus PDF-Encoding

        Args:
            text: Rohtext aus der PDF-Extraktion.

        Returns:
            Bereinigter Text.
        """
        import re

        # Mehrfache Leerzeilen auf maximal eine reduzieren
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Führende Leerzeichen am Zeilenanfang entfernen (PDF-Layout-Artefakte)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Soft-Hyphens (­) entfernen, die Wörter falsch trennen
        text = text.replace("\u00ad", "")

        return text.strip()

    def _create_chunks(
        self, pages: List[dict], source_filename: str
    ) -> List[Document]:
        """
        Zerlegt den Seitentext in Chunks und reichert sie mit Metadaten an.

        Die Metadaten sind entscheidend für die Quellen-Angabe in der Antwort:
        "Diese Information stammt aus Seite 7 von dokument.pdf"

        Args:
            pages: Liste von Seiten-Dicts aus _extract_pages().
            source_filename: Name der Quelldatei (für Metadaten).

        Returns:
            Liste von LangChain-Document-Objekten.
        """
        all_documents: List[Document] = []
        chunk_index = 0

        for page_data in pages:
            # Seiten-Text in Chunks aufteilen
            page_chunks = self.text_splitter.split_text(page_data["text"])

            for chunk_text in page_chunks:
                # Zu kurze Chunks sind oft Kopf-/Fußzeilen – überspringen
                if len(chunk_text.strip()) < 30:
                    continue

                doc = Document(
                    page_content=chunk_text,
                    metadata={
                        # Seite für Quellen-Angabe in der Antwort
                        "page": page_data["page_number"],
                        # Quelldatei für Multi-Dokument-Szenarien wichtig
                        "source": source_filename,
                        # Chunk-Index ermöglicht spätere Sortierung/Deduplizierung
                        "chunk_index": chunk_index,
                        # Zeichenanzahl hilft beim Debuggen ungleichmäßiger Chunks
                        "chunk_length": len(chunk_text),
                    },
                )
                all_documents.append(doc)
                chunk_index += 1

        return all_documents

    def get_stats(self, documents: List[Document]) -> dict:
        """
        Berechnet Statistiken über die erstellten Chunks.

        Nützlich für Debugging und zum Verstehen der Dokumentstruktur.

        Args:
            documents: Liste der erstellten Documents.

        Returns:
            Dict mit Statistiken (Anzahl, Durchschnittslänge, etc.).
        """
        if not documents:
            return {"count": 0}

        lengths = [len(doc.page_content) for doc in documents]
        pages = [doc.metadata.get("page", 0) for doc in documents]

        return {
            "count": len(documents),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
            "unique_pages": len(set(pages)),
        }
