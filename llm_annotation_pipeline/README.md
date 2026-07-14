# LLM annotation evaluation pipeline

This pipeline annotates every post in the two fixed trial threads and every post in two deterministically sampled threads from each of 4chan, Gab, Stormfront, and Vanguard.

It uses the matching prompt in `prompts/`, OpenAI structured outputs, strict Pydantic validation, deterministic sampling, append-only JSONL results, a selection manifest, retries, and item-level resume support.

## API key

Create a local `.env` file (ignored by Git) or export the variable:

```text
OPENAI_API_KEY=your_key_here
```

Never add the key to source code or commit `.env`.

## Run

```bash
# Inspect selection and counts without API calls
uv run python -m llm_annotation_pipeline.runner --dry-run

# One paid request as an end-to-end smoke test
uv run python -m llm_annotation_pipeline.runner --limit 1

# Complete evaluation
uv run python -m llm_annotation_pipeline.runner

# Every post in every fringe-platform validation thread (no trial items)
uv run python -m llm_annotation_pipeline.runner --all-validation
```

The quality-first default is full `gpt-5.4` with `medium` reasoning. Override either setting with `--model MODEL_ID` or `--reasoning-effort LEVEL`.

Model and reasoning effort are included in output filenames. Re-running an identical configuration and seed resumes its existing results instead of mixing configurations or paying for completed items again. Outputs go to `results/` inside this pipeline folder.


`--all-validation` includes every post in every validation thread from 4chan, Gab, Stormfront, and Vanguard and excludes trial items. Each completed item is flushed and synced to disk, so after Ctrl-C the identical command safely resumes by skipping saved item IDs.
