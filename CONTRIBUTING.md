# Contributing

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
make install          # editable install with dev extras
make test             # suite with coverage
make run              # http://localhost:8000/docs
```

## Before a PR

- `make lint` · `make format` · `make typecheck` · `make test` all pass.
- New behaviour ships with tests. Coverage gate is **85%**.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).

## Architecture

Ports and adapters. The core (`SearchService`, `RagService`) depends on three
protocols:

| Port | Default adapter | Production swap |
|------|-----------------|-----------------|
| `EmbeddingProvider` | `HashingEmbedder` (offline) | `SentenceTransformerEmbedder` |
| `VectorStore` | `InMemoryVectorStore` (NumPy) | pgvector / FAISS |
| `LLMProvider` | `ExtractiveAnswerer` (offline) | Gemini / OpenAI adapter |

Add a capability by implementing the protocol and wiring it in
`app/api/deps.py` — the composition root. Nothing else changes.
