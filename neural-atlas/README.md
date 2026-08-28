# Neural Atlas

Neural Atlas is a local-first observability system for open-weight transformers. It links three views of the same model:

1. **Anatomy** — the whole transformer, its layers, attention blocks, MLPs, residual stream, and output head.
2. **Weight atlas** — every checkpoint tensor with semantic zoom from exact block statistics to one stored scalar, its raw bits, and its byte address.
3. **Functional imaging** — token-by-token inference traces containing hidden-state changes, attention, MLP activity, logits, and exact MLP weight contributions.

The default model is `EleutherAI/pythia-70m-deduped`. Model files are downloaded from Hugging Face into the normal local cache and are never committed to this repository.

## Quick start

Requirements:

- Python 3.11 or newer
- Node.js 20.19 or newer, or Node.js 22.12 or newer
- A current WebGL-capable browser

```bash
cd neural-atlas
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

cd web
npm ci
npm run build
cd ..

neural-atlas
```

Open `http://127.0.0.1:8000`.

The first model load downloads the configured checkpoint. CPU execution is supported. Apple Silicon uses MPS automatically when available, and CUDA is used automatically on supported systems.

## Development mode

Run the API:

```bash
cd neural-atlas
source .venv/bin/activate
uvicorn neural_atlas.app:app --reload --host 127.0.0.1 --port 8000
```

Run the Vite interface in another terminal:

```bash
cd neural-atlas/web
npm ci
npm run dev
```

Open `http://127.0.0.1:5174`. Vite proxies HTTP and WebSocket requests under `/api` to the backend.

## Docker

```bash
cd neural-atlas
docker compose up --build
```

The Hugging Face cache is stored in a named Docker volume so model files survive container rebuilds.

## Configuration

Environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `NEURAL_ATLAS_MODEL_ID` | `EleutherAI/pythia-70m-deduped` | Hugging Face model ID or local model directory |
| `NEURAL_ATLAS_REVISION` | `main` | Model revision |
| `NEURAL_ATLAS_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `NEURAL_ATLAS_DTYPE` | `auto` | `auto`, `float32`, `float16`, or `bfloat16` |
| `NEURAL_ATLAS_HOST` | `127.0.0.1` | API bind address |
| `NEURAL_ATLAS_PORT` | `8000` | API port |
| `NEURAL_ATLAS_MAX_TILE_VALUES` | `10000000` | Maximum source values scanned by one tile request |

A local directory containing a Hugging Face-compatible model can be supplied as `NEURAL_ATLAS_MODEL_ID`. The weight atlas requires safetensors checkpoint files.

## Product workflow

### Whole-model anatomy

Select a layer in the Three.js scene to inspect its live residual, attention, and MLP measurements. The layout is schematic and stable: depth represents transformer order, vertical position separates module families, and brightness represents the currently selected trace measurement.

### Complete weight atlas

1. Load or index the model.
2. Search for a tensor.
3. Open its heatmap.
4. Click a summarized cell to zoom into that exact tensor region.
5. Continue until one displayed cell equals one stored scalar.
6. Select the scalar to view its tensor indices, value, dtype, raw bytes, raw bits, source file, and absolute file offset.

For tensors of rank greater than two, all leading dimensions are flattened into matrix rows while the final tensor dimension remains the matrix column. The inspector always returns the original multidimensional indices.

### Live inference

Enter a prompt and start a trace. Every generated token records:

- the exact input and selected output token IDs;
- final top-token probabilities;
- per-layer residual norms, deltas, means, standard deviations, and strongest dimensions;
- selected high-weight attention edges for the current token;
- strongest activated MLP channels;
- optional per-layer logit-lens candidates;
- latency for the forward pass.

The timeline can be scrubbed after generation. Selecting a layer synchronizes the 3D anatomy, attention view, activation table, and contribution controls.

### Exact contribution microscope

For GPT-NeoX MLPs, Neural Atlas captures the real activation vector entering `dense_4h_to_h`. Select a generation step, layer, and output channel to compute

```text
contribution[j] = weight[output_channel, j] × activation[j]
```

The response reports the exact weight, activation, product, total weighted sum, bias, and reconstructed linear output. This is a measured local decomposition of that linear operation; it is not a complete causal explanation of the final answer.

## API

Key endpoints:

- `GET /api/health`
- `POST /api/model/load`
- `GET /api/model`
- `POST /api/checkpoint/index`
- `GET /api/tensors`
- `GET /api/tensor`
- `GET /api/tile`
- `GET /api/scalar`
- `POST /api/contribution`
- `WS /api/trace/ws`

Interactive OpenAPI documentation is available at `/docs`.

## Verification

```bash
cd neural-atlas
source .venv/bin/activate
pytest

cd web
npm ci
npm run build
```

The safetensors tests create deterministic temporary checkpoints and verify tensor metadata, flattened matrix addressing, tile statistics, exact scalar values, raw bytes, bit patterns, and absolute file offsets. API tests use a temporary checkpoint and do not download the default model.

## Exactness boundary

- Checkpoint values and byte addresses come directly from safetensors headers and payload bytes.
- Weight tiles are exact for the requested region; requests that exceed the configured source-value limit are rejected rather than silently sampled.
- The live trace records real model outputs and activations.
- The anatomical arrangement is schematic.
- Attention is a measured operation, not automatically a causal explanation.
- The logit lens applies the final decoder normalization and output projection to intermediate hidden states; it is an interpretive probe.
- A single weight contribution describes one multiplication inside one linear operation, not the complete reason for a generated token.
