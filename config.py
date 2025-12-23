"""
config.py – Zentrale Konfiguration für das RAG-System.

Alle konfigurierbaren Werte sind hier gesammelt, damit:
  1. Keine "magic numbers" im Code verstreut sind
  2. Anpassungen schnell möglich sind (z.B. Chunk-Größe optimieren)
  3. API-Keys sicher aus der Umgebung geladen werden – NIE im Code hardcoden!
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env-Datei laden, falls vorhanden (überschreibt nicht bereits gesetzte Env-Vars)
load_dotenv()

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------

# Basisverzeichnis des Projekts (dieses config.py liegt im Root)
BASE_DIR = Path(__file__).parent

# Verzeichnis, in dem ChromaDB seine Daten persistent speichert
# Persistent = die Datenbank bleibt nach Programmende erhalten → kein erneutes
# Indexieren bei jedem Start
CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

# ---------------------------------------------------------------------------
# API-Keys (immer aus Umgebungsvariablen – nie hardcoden!)
# ---------------------------------------------------------------------------

# OpenAI-Key wird aus der .env-Datei oder aus der Shell-Umgebung gelesen
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# Embedding-Modell
# ---------------------------------------------------------------------------

# sentence-transformers/all-MiniLM-L6-v2 ist ein ausgewogenes Modell:
#   - 384-dimensionale Vektoren → klein und schnell
#   - Sehr gute Qualität für semantische Suche
#   - Komplett kostenlos & lokal ausführbar (kein API-Call nötig)
# Alternativ: "text-embedding-ada-002" von OpenAI (kostenpflichtig, aber stärker)
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Text-Splitting (Chunking)
# ---------------------------------------------------------------------------

# 500 Token ≈ ~375 Wörter. Diese Größe ist ein bewährter Kompromiss:
#   - Zu kleine Chunks → verlieren Kontext (ein Satz allein ist oft sinnlos)
#   - Zu große Chunks → überschreiten das LLM-Kontextfenster und sind weniger präzise
CHUNK_SIZE: int = 500

# Overlap verhindert, dass Informationen an Chunk-Grenzen verloren gehen.
# Wenn ein wichtiger Satz genau auf der Grenze liegt, erscheint er in BEIDEN Chunks.
CHUNK_OVERLAP: int = 50

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

# Wie viele der ähnlichsten Chunks werden für die Antwort herangezogen?
# K=3 ist ein guter Startpunkt: genug Kontext, aber nicht zu viel "Rauschen"
TOP_K_RESULTS: int = 3

# Similarity-Schwellwert: Chunks mit Score < MIN_SIMILARITY werden verworfen.
# ChromaDB gibt Distanzwerte zurück; wir rechnen sie in Ähnlichkeit um (1 - Distanz).
MIN_SIMILARITY: float = 0.3

# ---------------------------------------------------------------------------
# LLM-Konfiguration
# ---------------------------------------------------------------------------

# GPT-3.5-turbo ist für RAG-Anwendungen sehr gut geeignet:
#   - Günstig (ca. 1/20 des GPT-4-Preises)
#   - 16k-Token-Kontextfenster → reicht für mehrere Chunks + Antwort
#   - Schnell genug für interaktive Nutzung
# Wechsle zu "gpt-4o" für komplexere Dokumente oder mehrstufige Logik.
LLM_MODEL: str = "gpt-3.5-turbo"

# Temperature = 0 → deterministisch, ideal für faktenbasierte Antworten.
# Höhere Werte (0.7+) machen Antworten kreativer, aber weniger verlässlich.
LLM_TEMPERATURE: float = 0.0

# Maximale Anzahl der generierten Tokens pro Antwort
LLM_MAX_TOKENS: int = 1024

# ---------------------------------------------------------------------------
# ChromaDB-Kollektion
# ---------------------------------------------------------------------------

# Name der Kollektion in ChromaDB (ähnlich wie ein Tabellenname in SQL)
COLLECTION_NAME: str = "rag_documents"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

# Logging-Level: DEBUG gibt sehr detaillierte Infos, INFO ist für normale Nutzung
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%H:%M:%S"
