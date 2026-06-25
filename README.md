# UCSB Counterspeech Annotation Project

This repository contains a Flask annotation interface and supporting analysis scripts for counterspeech and hate-speech classification experiments.

## Quick Start

```bash
uv sync
uv run flask --app annotation_app.app run
```

Open the local Flask URL and read the instructions page before starting annotation. The app loads annotation items from `annotation_app/data/items.json` and writes submissions to dated CSV and JSONL files in `annotation_storage/`.

If you are not using `uv`, install the Python dependencies with:

```bash
pip install -r requirements.txt
python annotation_app/app.py
```

## Project Layout

```text
.
├── annotation_app/                 # Flask annotation app
│   ├── app.py                     # Routes and annotation save logic
│   ├── data/items.json            # Active annotation queue consumed by the app
│   ├── templates/                 # HTML templates for instructions and annotation UI
│   └── static/media/              # Public instruction video and poster assets
├── data/
│   ├── sample/                    # Sample source dataset used by scripts
│   ├── reference/                 # Converted samples and derived reference files
│   └── diagnosis/                 # Diagnostic reports
├── annotation_storage/            # Generated annotation exports
├── scripts/                       # Reusable data, diagnostic, and classification utilities
├── gepa_prompt_optimization/      # DSPy / prompt optimization experiments
├── results_thread_context_improved/ # Model classification outputs
└── old_runs/                      # Historical experiments kept for provenance
```

## Common Tasks

Generate the annotation queue from the raw thread export:

```bash
python scripts/convert_threads_to_items.py
```

Run dataset diagnostics:

```bash
python scripts/diagnose_data.py
```

Run the thread-context classifier:

```bash
python scripts/classify_thread_context.py
```

Extract thread features:

```bash
python scripts/extract_features.py
```

## Data Conventions

- Keep the sample source files in `data/sample/`.
- Keep the active app input at `annotation_app/data/items.json`.
- Keep generated annotation exports in `annotation_storage/`.
- Keep one-off or historical experiments in `old_runs/` instead of the project root.
- Avoid committing local caches, virtual environments, or secret files.

## Environment Variables

Some classification and optimization scripts require `OPENAI_API_KEY` in your environment or in a local `.env` file. The Flask annotation app does not need an API key.
