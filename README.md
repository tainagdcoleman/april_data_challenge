# Dataset RAG Chatbot

A Jupyter-notebook data challenge: index metadata about a handful of datasets into a vector database and expose a small chatbot that answers questions by retrieving the relevant metadata.

- **Chat + embedding models** are configured in `models.yaml` and called through the OpenAI SDK, so you can swap between NRP-managed models (`gemma`, `qwen3`, `glm-4.7`, `qwen3-embedding`) or any OpenAI-compatible endpoint by changing one key in the notebook.
- **Datasets** come from either the NDP CKAN catalog (by slug/UUID) or local files already on disk.
- **Vector store** is ChromaDB (file-based, persisted under `./chroma_db/`), fronted by a small factory so you can add other backends later.

## Layout

```
.
├── dataset_rag.ipynb   # main notebook — run top to bottom
├── models.yaml         # chat + embedding model configs (uses ${ENV_VAR})
├── get_model.py        # load_models_config, call_model, call_embedding
├── base_agent.py       # BaseAgent class (history + LLM helpers)
├── vector_stores.py    # ChromaStore + make_store(...) factory
├── requirements.txt
└── .gitignore
```

## Setup

1. Create a `.env` from the template and fill in your NRP API key:
   ```bash
   cp .env.example .env
   # edit .env → NRP_API_KEY=...
   ```
2. Install dependencies (or run the `%pip install` cell in §0 of the notebook):
   ```bash
   pip install -r requirements.txt
   ```
3. Open `dataset_rag.ipynb` and run the cells top to bottom.

## Swapping models

The notebook has two knobs near the top (§2):

```python
CHAT_MODEL_KEY = 'gemma'              # any entry in models.yaml with task: chat
EMBED_MODEL_KEY = 'qwen3-embedding'   # task: embed; None = local sentence-transformers
```

Both keys must match entries in [`models.yaml`](models.yaml). To add a new model, copy an existing block and point `base_url` at the new provider — `call_model` / `call_embedding` use the OpenAI SDK and accept any compatible endpoint.

Reasoning models (like `qwen3`) work transparently: `call_model` defaults `max_tokens=2048` so they have room to finish thinking *and* produce a final answer.

## Dataset sources

§3 of the notebook has a `DATASET_SOURCE` toggle:

- `'ndp'` — fetch fresh metadata from the NDP CKAN catalog. Fill `DATASET_IDS` with slugs/UUIDs (or leave empty to use the first 3 datasets in the catalog).
- `'local'` — load metadata from a directory in `DATASETS_PATH`. Two accepted layouts:
  - flat directory of `.json` files (one CKAN-style metadata dict per file), or
  - directory of subfolders, each containing `metadata.json` / `metadata.yaml`. If a subfolder has no metadata file, one is auto-synthesized from the folder name and file listing.

## Environment variables

| var | purpose |
|---|---|
| `NRP_API_KEY` | required for NRP-managed chat/embedding models |

