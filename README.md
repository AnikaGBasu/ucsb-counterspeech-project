# UCSB Counterspeech Annotation Project

This repository contains a Flask annotation interface and supporting analysis scripts for counterspeech and hate-speech classification experiments.

## Quick Start

```bash
uv sync
uv run flask --app annotation_app.app run
```

Open the local Flask URL and read the instructions page before starting annotation. After consent, annotators choose between the trial annotation and the full annotation. The trial task loads the fixed trial set from `annotation_app/data/items.json`; dynamic assignment applies only to the full task. The full task dynamically assigns one validation thread at a time from `data/validation_set_jsons/`, restores saved progress for returning annotators, and requires completing the assigned thread for payment. Submissions are written to dated CSV and JSONL files in `annotation_app/annotation_storage/`.

Gab advertisement placeholders (`gab-ad-comment`) are discarded during validation-data conversion, so they are never presented for human or LLM annotation.

To share the local app through ngrok, start Flask first, then run this in another terminal and share the forwarding URL. Annotators should open the forwarding URL at `/`, read the instructions, save consent, and choose either Trial Annotation or Full Annotation. Full Annotation assigns one validation thread at a time:

```bash
ngrok http 5000
```

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
│   ├── data/items.json            # Trial annotation queue consumed by the app
│   ├── templates/                 # HTML templates for instructions and annotation UI
│   ├── static/media/              # Public instruction video and poster assets
│   └── annotation_storage/        # Generated annotation exports
├── data/
│   ├── sample/                    # Sample source dataset used by scripts
│   ├── reference/                 # Converted samples and derived reference files
│   ├── diagnosis/                 # Diagnostic reports
│   └── validation_set_jsons/      # Validation datasets used by the full annotation task
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

Reproduce the manually curated 50 hate-speech / 50 non-hate original-post annotation set from `data/full_data/`:

```bash
python scripts/select_original_posts_for_annotation.py
```

The script first writes an unlabeled outermost-post review pool to `data/annotation_selection/original_post_review_pool.csv`, then applies `data/annotation_selection/manual_curation_plan.json`. It intentionally does not keyword-sample or auto-label posts. To make the selected set the live Flask annotation queue, run:

```bash
python scripts/select_original_posts_for_annotation.py --install
```

Extract thread features:

```bash
python scripts/extract_features.py
```

Check inter-annotator agreement for one or more annotation exports:

```bash
python scripts/check_inter_annotator_agreement.py annotation_app/annotation_storage/annotations_20260709.csv
```

The agreement script compares the latest saved row for each annotator/item pair, reports pairwise percent agreement and mean pairwise Cohen kappa by label field, and prints example disagreements for inspection.

## Data Conventions

- Keep the sample source files in `data/sample/`.
- Keep the trial app input at `annotation_app/data/items.json`.
- Keep validation datasets for the full annotation task in `data/validation_set_jsons/`.
- Keep generated annotation exports in `annotation_app/annotation_storage/`.
- Keep one-off or historical experiments in `old_runs/` instead of the project root.
- Avoid committing local caches, virtual environments, or secret files.

## Environment Variables

Some classification and optimization scripts require `OPENAI_API_KEY` in your environment or in a local `.env` file. The Flask annotation app does not need an API key.
