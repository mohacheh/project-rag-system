"""
utils.py – Querschnittsfunktionen (Logging, Ausgabe, Validierung).

Das Auslagern von Hilfsfunktionen in eine eigene Datei folgt dem
Single-Responsibility-Prinzip: jede Datei hat eine klar definierte Aufgabe.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import config


def setup_logging(log_level: Optional[str] = None) -> logging.Logger:
    """
    Konfiguriert das zentrale Logging für die gesamte Anwendung.

    Wir verwenden logging statt print(), weil:
    - Log-Level ermöglichen selektive Ausgabe (DEBUG nur wenn nötig)
    - Timestamps und Modulnamen helfen beim Debuggen
    - Ausgaben können einfach in Dateien umgeleitet werden

    Args:
        log_level: Optionaler Override für das Log-Level (z.B. "DEBUG").
                   Standard kommt aus config.py.

    Returns:
        Konfigurierter Root-Logger.
    """
    level = getattr(logging, (log_level or config.LOG_LEVEL).upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format=config.LOG_FORMAT,
        datefmt=config.LOG_DATE_FORMAT,
        stream=sys.stdout,
    )

    # Externe Bibliotheken auf WARNING setzen, damit ihre Debug-Logs uns nicht fluten
    for noisy_lib in ("chromadb", "httpx", "httpcore", "openai", "sentence_transformers"):
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    return logging.getLogger(__name__)


def validate_pdf_path(path: str) -> Path:
    """
    Überprüft, ob eine PDF-Datei existiert und lesbar ist.

    Frühzeitige Validierung (fail-fast) verhindert, dass der Nutzer erst
    nach dem langsamen Embedding-Prozess einen Fehler bemerkt.

    Args:
        path: Dateipfad zur PDF-Datei.

    Returns:
        Validiertes Path-Objekt.

    Raises:
        FileNotFoundError: Wenn die Datei nicht existiert.
        ValueError: Wenn die Datei keine PDF ist.
    """
    pdf_path = Path(path).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF nicht gefunden: {pdf_path}\n"
            "Bitte prüfe den Pfad und stelle sicher, dass die Datei existiert."
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Erwartet eine .pdf-Datei, erhalten: {pdf_path.suffix}\n"
            "Nur PDF-Dateien werden unterstützt."
        )

    return pdf_path


def validate_api_key() -> None:
    """
    Überprüft, ob der OpenAI API-Key gesetzt ist.

    Raises:
        EnvironmentError: Wenn der Key fehlt oder leer ist.
    """
    if not config.OPENAI_API_KEY:
        raise EnvironmentError(
            "OpenAI API-Key nicht gefunden!\n"
            "Lösung: Erstelle eine .env-Datei mit:\n"
            "  OPENAI_API_KEY=sk-...\n"
            "Oder setze die Umgebungsvariable: export OPENAI_API_KEY=sk-..."
        )


def format_separator(char: str = "─", width: int = 50) -> str:
    """
    Gibt eine formatierte Trennlinie zurück.

    Args:
        char: Zeichen für die Trennlinie.
        width: Breite der Linie.

    Returns:
        Trennlinien-String.
    """
    return char * width


def print_header() -> None:
    """Gibt den Anwendungs-Header in der Konsole aus."""
    print("\n" + "=" * 50)
    print("  🤖  RAG-System – Dokumenten-KI")
    print("  Powered by LangChain + ChromaDB + OpenAI")
    print("=" * 50 + "\n")


def format_similarity_score(score: float) -> str:
    """
    Konvertiert einen Similarity-Score in einen lesbaren String mit Emoji.

    Wir wandeln Distanzwerte in intuitive Ähnlichkeitswerte um:
    1.0 = perfekter Treffer, 0.0 = komplett unterschiedlich.

    Args:
        score: Similarity-Score zwischen 0 und 1.

    Returns:
        Formatierter Score-String mit visueller Bewertung.
    """
    if score >= 0.85:
        quality = "🟢 Sehr hoch"
    elif score >= 0.70:
        quality = "🟡 Hoch"
    elif score >= 0.50:
        quality = "🟠 Mittel"
    else:
        quality = "🔴 Niedrig"

    return f"{score:.3f} ({quality})"
