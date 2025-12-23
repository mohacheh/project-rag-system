🧠 PDF-Inquire: Professional RAG SystemEin modulares RAG-System (Retrieval-Augmented Generation), das es ermöglicht, mit lokalen PDF-Dokumenten in natürlicher Sprache zu chatten. Optimiert für Präzision, Kosteneffizienz und Datenschutz durch lokale Embeddings.📋 InhaltsverzeichnisWas ist RAG?ArchitekturKey FeaturesInstallationTechnische EntscheidungenBeispiel-OutputProjektstrukturRoadmap💡 Was ist RAG?RAG löst das Problem der "Halluzinationen" und veralteten Daten bei LLMs. Anstatt sich auf das statische Wissen des Trainings zu verlassen, fungiert das System als digitaler Bibliothekar:Ingestion: PDFs werden in semantische Einheiten (Chunks) zerlegt.Retrieval: Bei einer Frage sucht das System in Millisekunden die relevantesten Textstellen.Augmentation: Das LLM erhält die Frage zusammen mit dem gefundenen Kontext.Generation: Die Antwort basiert ausschließlich auf den bereitgestellten Fakten.🏗 ArchitekturDas System ist in zwei Pipelines unterteilt:1. Indexierungs-Pipeline (Offline)Extraktion: PyMuPDF extrahiert Text & Metadaten.Chunking: RecursiveCharacterTextSplitter bewahrt semantische Zusammenhänge.Embedding: Lokales sentence-transformers/all-MiniLM-L6-v2 (384 Dimensionen).Storage: ChromaDB als persistenter Vektor-Store.2. Query-Pipeline (Online)Semantic Search: Wandelt die Nutzerfrage in einen Vektor um und findet Top-k Übereinstimmungen.Prompt Engineering: Ein spezialisierter System-Prompt verhindert "Erfindungen" des LLMs.Response Generation: GPT-3.5-Turbo generiert die Antwort mit präzisen Quellenangaben.🚀 Key FeaturesZero-Cost Embeddings: Verwendet lokale HuggingFace-Modelle – spart Kosten und schützt Daten.Hybrid-Metadata: Jede Antwort enthält Quellenangaben (Seite & Dateiname) für maximale Transparenz.Smart-Chunking: 500 Token Fenster mit 10% Overlap verhindert Informationsverlust an Schnittstellen.Persistence: Einmal indexierte Dokumente sind sofort wieder verfügbar (SQLite-Backend via ChromaDB).Cost-Control: Live-Tracking der verbrauchten OpenAI-Tokens pro Session.🛠 InstallationVoraussetzungenPython 3.10+OpenAI API-KeyBash# Repo & Environment
git clone https://github.com/dein-username/rag-system.git
cd rag-system
python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate

# Dependencies
pip install -r requirements.txt

# Config
cp .env.example .env
# Trage deinen OPENAI_API_KEY in die .env ein
VerwendungBash# Start mit einer PDF
python main.py --pdf ./manual.pdf

# Fortgeschritten: Mehr Kontext für komplexe Fragen
python main.py --pdf ./report.pdf --top-k 5 --reset-db
⚖️ Technische EntscheidungenKomponenteWahlGrundLLMGPT-3.5-TurboOptimales Preis-Leistungs-Verhältnis für Extraktionsaufgaben.EmbeddingsMiniLM (L6)Extrem schnell, lokal ausführbar, geringer RAM-Verbrauch (~200MB).Vector DBChromaDBNative Metadaten-Filterung und einfache Persistenz im Vergleich zu FAISS.ParserPyMuPDFDeutlich höhere Geschwindigkeit als PyPDF2 bei komplexen Layouts.📊 Beispiel-OutputPlaintextDu: Welche Kündigungsfrist gilt im ersten Jahr?

🤖 Antwort:
Gemäß Abschnitt 4.2 Ihres Arbeitsvertrags beträgt die Kündigungsfrist 
innerhalb der Probezeit (erste 6 Monate) zwei Wochen. Nach Ablauf 
der Probezeit gilt im ersten Beschäftigungsjahr eine Frist von 
einem Monat zum Monatsende.

📎 Quellen: Seite 4, Seite 12
💰 Kosten dieser Anfrage: $0.0008