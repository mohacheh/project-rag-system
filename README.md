# 📑 PDF-Inquire: Professional RAG System

**PDF-Inquire** ist ein modulares RAG-System (Retrieval-Augmented Generation), das präzise Antworten auf Basis deiner lokalen Dokumente liefert. Durch die Kombination von **lokalen Embeddings** und **Cloud-basierten LLMs** bietet es die perfekte Balance zwischen Datenschutz, Geschwindigkeit und Kosteneffizienz.

----

## 💡 Was ist RAG?

Standard-LLMs neigen zu "Halluzinationen", wenn sie über spezifische oder private Daten abgefragt werden. RAG löst dieses Problem, indem es das Modell in einen **digitalen Bibliothekar** verwandelt:



* **Ingestion:** PDFs werden in semantische Fragmente (Chunks) zerlegt.
* **Retrieval:** Das System findet in Millisekunden die relevantesten Stellen für deine Frage.
* **Augmentation:** Das LLM erhält die Frage zusammen mit dem exakten Kontext.
* **Generation:** Die Antwort basiert faktentreu auf den bereitgestellten Daten.

---

## 🏗 Architektur

Das System trennt strikt zwischen Datenvorbereitung und Abfrage-Logik:

### 1. Indexierungs-Pipeline (Offline)
* **Extraktion:** `PyMuPDF` extrahiert Text und Metadaten (Seitenzahlen, Dateinamen).
* **Chunking:** `RecursiveCharacterTextSplitter` nutzt ein Fenster von 500 Token mit 10% Overlap.
* **Embedding:** Lokale Ausführung via `sentence-transformers/all-MiniLM-L6-v2`.
* **Storage:** `ChromaDB` (persistentes SQLite-Backend).

### 2. Query-Pipeline (Online)
* **Semantic Search:** Vektorbasiert Suche nach den Top-k Übereinstimmungen.
* **Prompt Engineering:** Spezialisierte System-Prompts erzwingen die Nutzung des Kontexts.
* **Response Generation:** `GPT-3.5-Turbo` (oder neuer) liefert die finale Antwort inklusive Quellenangaben.

---

## 🚀 Key Features

| Feature | Beschreibung |
| :--- | :--- |
| **Zero-Cost Embeddings** | Lokale HuggingFace-Modelle sparen API-Kosten und erhöhen den Datenschutz. |
| **Hybrid-Metadata** | Jede Antwort nennt Seite & Dateiname zur Verifizierung. |
| **Smart-Chunking** | Verhindert Informationsverlust durch intelligenten Text-Overlap. |
| **Persistence** | Einmal indexierte Dokumente bleiben dauerhaft gespeichert. |
| **Token Tracking** | Transparente Übersicht der OpenAI-Kosten pro Session. |

---

## 🛠 Installation & Setup

### Voraussetzungen
* Python 3.10 oder höher
* OpenAI API-Key

### Schritt-für-Schritt

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/dein-username/rag-system.git](https://github.com/dein-username/rag-system.git)
    cd rag-system
    ```

2.  **Virtuelle Umgebung einrichten:**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Umgebungsvariablen konfigurieren:**
    Erstelle eine `.env` Datei im Hauptverzeichnis:
    ```bash
    OPENAI_API_KEY=dein_key_hier
    DB_PATH=./chroma_db
    DOCS_PATH=./data
    ```

---

## 📂 Projektstruktur

```text
rag-system/
├── data/               # Deine PDFs
├── chroma_db/          # Persistenter Vektorspeicher
├── src/
│   ├── ingestion.py    # PDF Processing & Embedding
│   ├── retrieval.py    # Suche & RAG Logik
│   └── app.py          # CLI oder UI Interface
├── .env                # API Keys (nicht einchecken!)
├── .gitignore          # Schließt venv, .env und DB aus
├── requirements.txt
└── README.md
