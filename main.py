"""
main.py – Einstiegspunkt des RAG-Systems.

Dieses Modul orchestriert alle Komponenten:
  PDFProcessor → VectorStore → Retriever → LLMChain

Das ist das Fassaden-Muster (Facade Pattern):
main.py kennt alle Komponenten, aber die Komponenten kennen sich nicht gegenseitig.
Das macht sie einzeln testbar und austauschbar.

Verwendung:
  python main.py --pdf ./dokumente/beispiel.pdf
  python main.py --pdf ./dokumente/beispiel.pdf --top-k 5 --debug
"""

import argparse
import logging
import sys
from pathlib import Path

import config
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from retriever import Retriever
from llm_chain import LLMChain
from utils import (
    setup_logging,
    validate_pdf_path,
    validate_api_key,
    print_header,
    format_separator,
)

logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Verarbeitet Kommandozeilen-Argumente.

    argparse statt sys.argv direkt, weil:
    - Automatische --help-Generierung
    - Typ-Validierung
    - Übersichtlicherer Code

    Returns:
        Geparste Argumente als Namespace-Objekt.
    """
    parser = argparse.ArgumentParser(
        description="RAG-System: Stelle Fragen an PDF-Dokumente",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python main.py --pdf ./dokumente/bericht.pdf
  python main.py --pdf ./handbuch.pdf --top-k 5
  python main.py --pdf ./vertrag.pdf --debug
        """,
    )

    parser.add_argument(
        "--pdf",
        type=str,
        required=True,
        help="Pfad zur PDF-Datei (z.B. ./dokumente/meine_datei.pdf)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=config.TOP_K_RESULTS,
        help=f"Anzahl der zu holenden Kontext-Chunks (Standard: {config.TOP_K_RESULTS})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Aktiviert detailliertes Debug-Logging",
    )
    parser.add_argument(
        "--reset-db",
        action="store_true",
        help="Löscht die bestehende Vektordatenbank vor dem Indexieren",
    )

    return parser.parse_args()


def setup_rag_pipeline(
    pdf_path: Path, top_k: int, reset_db: bool
) -> tuple[Retriever, LLMChain]:
    """
    Initialisiert und konfiguriert die vollständige RAG-Pipeline.

    Diese Funktion folgt dem "Setup → Use → Teardown"-Muster:
    Alle schweren Initialisierungen (Modell laden, DB öffnen) hier zentralisiert.

    Args:
        pdf_path: Validierter Pfad zur PDF-Datei.
        top_k: Anzahl der Kontext-Chunks pro Anfrage.
        reset_db: Ob die DB vor dem Indexieren geleert werden soll.

    Returns:
        Tuple aus (Retriever, LLMChain) – ready to use.
    """
    # --- Schritt 1: PDF verarbeiten und chunken ---
    print(f"\n📄 Lade PDF: {pdf_path.name}")
    processor = PDFProcessor()
    documents = processor.load_and_split(pdf_path)

    # Chunk-Statistiken anzeigen (hilft beim Debuggen unerwarteter Ergebnisse)
    stats = processor.get_stats(documents)
    print(
        f"   Durchschnittliche Chunk-Länge: {stats['avg_length']:.0f} Zeichen "
        f"| Seiten: {stats['unique_pages']}"
    )

    # --- Schritt 2: Vektordatenbank initialisieren ---
    print("\n🗄️  Initialisiere Vektordatenbank...")
    vector_store = VectorStore()

    # Optional: Bestehende DB löschen (z.B. nach Änderungen am PDF)
    if reset_db:
        print("  ⚠️  Bestehende Datenbank wird geleert...")
        import shutil
        if Path(config.CHROMA_PERSIST_DIR).exists():
            shutil.rmtree(config.CHROMA_PERSIST_DIR)
        vector_store = VectorStore()  # Neu initialisieren nach dem Löschen

    # --- Schritt 3: Embeddings berechnen und speichern ---
    print(f"\n🔢 Berechne Embeddings für {len(documents)} Chunks...")
    added = vector_store.add_documents(documents)

    total_in_db = vector_store.get_document_count()
    print(f"   ✅ {total_in_db} Chunks in der Datenbank ({added} neu hinzugefügt)")

    # --- Schritt 4: Retriever und LLM-Chain aufbauen ---
    retriever = Retriever(vector_store=vector_store, top_k=top_k)
    llm_chain = LLMChain()

    return retriever, llm_chain


def interactive_chat_loop(retriever: Retriever, llm_chain: LLMChain) -> None:
    """
    Startet die interaktive Frage-Antwort-Schleife.

    Die Chat-Loop läuft bis der Nutzer "exit" oder "quit" eingibt
    oder Ctrl+C drückt (KeyboardInterrupt).

    Args:
        retriever: Konfigurierter Retriever für Dokumenten-Suche.
        llm_chain: Konfigurierte LLM-Chain für Antwortgenerierung.
    """
    print("\n" + format_separator("─", 50))
    print("💬 RAG-System bereit. Stelle deine Fragen!")
    print("   (Tippe 'exit' zum Beenden, 'stats' für Token-Statistiken)")
    print(format_separator("─", 50))

    question_count = 0

    while True:
        try:
            # Nutzereingabe
            print()
            user_input = input("Du: ").strip()

            # Leere Eingaben ignorieren
            if not user_input:
                continue

            # Exit-Kommando
            if user_input.lower() in ("exit", "quit", "bye", "beenden"):
                _print_session_summary(llm_chain, question_count)
                print("\n👋 Auf Wiedersehen!")
                break

            # Stats-Kommando
            if user_input.lower() == "stats":
                _print_session_stats(llm_chain)
                continue

            question_count += 1

            # --- RAG-Pipeline ausführen ---

            # 1. Relevante Chunks suchen
            retrieval_result = retriever.retrieve(user_input)

            # 2. Antwort vom LLM generieren
            print("\n🤖 Antwort wird generiert...")
            llm_response = llm_chain.generate_answer(retrieval_result)

            # 3. Antwort strukturiert ausgeben
            _print_answer(llm_response, retrieval_result)

        except KeyboardInterrupt:
            # Ctrl+C graceful abfangen
            _print_session_summary(llm_chain, question_count)
            print("\n\n👋 Abbruch per Ctrl+C. Auf Wiedersehen!")
            break

        except RuntimeError as e:
            # Bekannte Fehler (API, DB) benutzerfreundlich anzeigen
            print(f"\n❌ Fehler: {e}")
            logger.error(f"RuntimeError in Chat-Loop: {e}", exc_info=True)

        except Exception as e:
            # Unerwartete Fehler loggen und weitermachen
            print(f"\n❌ Unerwarteter Fehler: {e}")
            logger.error(f"Unerwarteter Fehler: {e}", exc_info=True)
            print("   Das System läuft weiter. Stelle eine neue Frage.")


def _print_answer(llm_response, retrieval_result) -> None:
    """
    Formatiert und gibt die LLM-Antwort aus.

    Trennt Antwort, Quellen und Token-Info visuell,
    damit Nutzer sofort die wichtigen Informationen sehen.
    """
    print("\n" + format_separator("─", 50))
    print(f"\n🤖 Antwort:\n{llm_response.answer}")

    if llm_response.sources:
        sources_str = ", ".join(llm_response.sources)
        print(f"\n📎 Quellen: {sources_str}")

    # Token-Info dezent anzeigen (für Kosten-Bewusstsein)
    if llm_response.total_tokens > 0:
        print(
            f"\n   [Token: {llm_response.total_tokens} | "
            f"≈ ${llm_response.estimated_cost_usd:.4f}]"
        )

    print(format_separator("─", 50))


def _print_session_stats(llm_chain: LLMChain) -> None:
    """Gibt die Token-Statistiken der aktuellen Session aus."""
    stats = llm_chain.get_session_stats()
    print(f"\n📊 Session-Statistiken:")
    print(f"   Modell: {stats['model']}")
    print(f"   Token gesamt: {stats['total_tokens']:,}")
    print(f"   Geschätzte Kosten: ${stats['estimated_cost_usd']:.4f}")


def _print_session_summary(llm_chain: LLMChain, question_count: int) -> None:
    """Gibt eine Zusammenfassung der Session aus."""
    stats = llm_chain.get_session_stats()
    print(f"\n\n📊 Session-Zusammenfassung:")
    print(f"   Fragen gestellt: {question_count}")
    print(f"   Token verbraucht: {stats['total_tokens']:,}")
    print(f"   Geschätzte Kosten: ${stats['estimated_cost_usd']:.4f}")


def main() -> int:
    """
    Hauptfunktion – Einstiegspunkt des Programms.

    Returns:
        Exit-Code (0 = Erfolg, 1 = Fehler).
    """
    args = parse_arguments()

    # Logging einrichten (vor allem anderen)
    setup_logging("DEBUG" if args.debug else None)

    print_header()

    # Frühzeitige Validierungen (fail-fast)
    try:
        pdf_path = validate_pdf_path(args.pdf)
        validate_api_key()
    except (FileNotFoundError, ValueError, EnvironmentError) as e:
        print(f"❌ Konfigurationsfehler: {e}")
        return 1

    # RAG-Pipeline aufbauen
    try:
        retriever, llm_chain = setup_rag_pipeline(
            pdf_path=pdf_path,
            top_k=args.top_k,
            reset_db=args.reset_db,
        )
    except Exception as e:
        print(f"❌ Fehler beim Setup: {e}")
        logger.error("Setup fehlgeschlagen", exc_info=True)
        return 1

    # Interaktive Chat-Schleife starten
    interactive_chat_loop(retriever, llm_chain)

    return 0


if __name__ == "__main__":
    sys.exit(main())
