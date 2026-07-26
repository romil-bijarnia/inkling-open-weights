# Inkling 3D Open Weights

Inkling 3D Open Weights is an interactive, real-data visualization of Thinking Machines Lab's published Inkling checkpoint. It maps the model's 66 decoder layers, 1,552 named tensors, routed and shared experts, exact sampled weight values, and the complete 200,058-entry tokenizer vocabulary into a searchable Three.js scene.

This is not a generated picture of a neural network. Architecture positions come from the published module and tensor structure. Every point in the Embeddings view represents one real learned input row from `model.llm.embed.weight`, and every selected-token relationship is an exact cosine comparison in the original 6,144-dimensional space.

## What the visualization shows

The **Architecture** view maps the complete decoder stack and all named checkpoint tensors. Search and inspection expose each tensor's name, shape, dtype, scalar count, byte count, source shard, and architectural role.

The **Experts** view opens a sparse layer's 256 routed experts and two shared experts. Gate-bias values control the displayed expert heights, but the visualization does not imply those values alone reveal a token's runtime route.

The **Embeddings** view includes all 200,058 tokenizer-backed rows. The cloud is a three-dimensional PCA projection for navigation; its 32 selected-token spokes come from exhaustive cosine comparisons against the entire vocabulary in the original 6,144-dimensional vectors. Token pieces such as `king` and `␠king` remain separate because they are different tokenizer entries.

The **Real values** view renders exact decoded BF16 or F32 samples as three-dimensional value surfaces. When the optional full embedding payload is installed, it can also display all 6,144 stored components of a selected token vector.

## Quick start

The checked-in data is sufficient for the complete all-token cloud, full-vocabulary search, and exact 32-neighbour relationships. Running the application itself does not require Python.

Prerequisites are Git, a current WebGL-capable browser, and Node.js `^20.19.0` or `>=22.12.0`.

```sh
git clone https://github.com/romil-bijarnia/inkling-open-weights.git
cd inkling-open-weights
npm ci
npm run dev
```

Open the local address printed by Vite, normally `http://127.0.0.1:5173`. To make the development server visible to other devices on your local network, use `npm run dev -- --host 0.0.0.0` and open the computer's LAN address from those devices.

## Optional full token vectors

GitHub does not store the 2,458,312,704-byte raw embedding payload. The public repository instead contains the complete vocabulary, 3D layout, group assignments, and exact global-neighbour graph. Download the raw BF16 rows only if you want the 6,144-component heatmap and Real values view for individual tokens.

```sh
npm run download:embeddings:full
npm run verify:embeddings:full
```

The downloader reads only the tokenizer-backed range of `model.llm.embed.weight` from the pinned Hugging Face shard. It is resumable, verifies known rows against the committed sample, and records a SHA-256 manifest. Allow about 3 GB of free disk space for the download. A production build made after the download can require roughly another 2.5 GB because Vite copies the payload into `dist/`.

If the payload is absent, the main visualization and exact token-neighbour relationships continue to work. The app identifies the optional download command when a raw per-dimension view is requested.

## Reproducible Python environment

Python is needed only to refresh, rebuild, or verify the derived checkpoint data. The repository includes a strict `requirements.txt` matching the versions used for the current data build. Python 3.13 is recommended.

On macOS or Linux:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Once activated, the virtual environment is used automatically by the `npm run` data commands because its `python3` executable appears first on the shell path.

## Validate and build

Validate the checkpoint metadata and the checked-in embedding products with:

```sh
npm run verify:data
npm run verify:embeddings
```

If the optional full payload has been downloaded, also run:

```sh
npm run verify:embeddings:full
```

Create and serve a production build locally with:

```sh
npm run build
npm run preview
```

The generated `dist/` directory is a static site. Any ordinary local HTTP server can host it; opening `index.html` directly through `file://` will not work because the app loads binary ranges and JSON assets over HTTP. For example:

```sh
python -m http.server 4173 --directory dist
```

Then open `http://127.0.0.1:4173`.

## Refresh the checkpoint data

The metadata refresh reads the public config, tensor index, all referenced safetensors headers, and small HTTP byte ranges for selected numerical slices:

```sh
python scripts/fetch_inkling_metadata.py
npm run verify:data
```

To reproduce the compact embedding sample, run:

```sh
npm run build:embeddings
npm run verify:embeddings
```

To rebuild the complete all-token PCA layout and exhaustive neighbour graph from the downloaded 2.46 GB matrix, run:

```sh
npm run download:embeddings:full
npm run build:embeddings:full
npm run verify:embeddings:full
```

The complete rebuild expands approximately 1.23 billion BF16 values and performs exhaustive all-vocabulary neighbour search. It is a heavy offline computation, not a normal installation step. The checked-in processed products exist so ordinary users do not need to repeat it.

## Repository layout

```text
data/                  Published metadata and browser-ready visualization data
scripts/               Download, build, and verification pipelines
src/main.js            Three.js scene, interaction, search, and inspectors
src/styles.css         Responsive visualization and project-menu styling
index.html             Application shell
requirements.txt       Exact Python data-tool dependencies
vite.config.js         Local development and static build configuration
```

Generated dependencies, local virtual environments, browser test output, production builds, and the 2.46 GB raw embedding matrix are deliberately excluded by `.gitignore`.

## Data provenance and honesty boundary

The snapshot comes from [`thinkingmachines/Inkling`](https://huggingface.co/thinkingmachines/Inkling), licensed Apache 2.0 and pinned at commit [`86b4d430ab871652a707666b89203a866888c5e5`](https://huggingface.co/thinkingmachines/Inkling/commit/86b4d430ab871652a707666b89203a866888c5e5). It records complete metadata for 1,552 named tensors across 109 safetensors files, accounting for 952,377,623,626 serialized scalars and approximately 1.905 TB of checkpoint weight data.

The repository does not contain all 1.905 TB of numerical checkpoint weights. Architecture-wide numerical views are exact, reproducible byte-range samples from named tensors. The embedding products cover all 200,058 tokenizer entries and exclude only 966 padded rows with no tokenizer entry.

The 3D embedding positions are a PCA projection and therefore cannot preserve every relationship from 6,144 dimensions. Exact neighbour scores shown after selection are computed in the original 6,144-dimensional vectors, not from visual distance in the projected cloud. These are static input-token embeddings, not contextual activations, facts, or a complete inventory of what the model knows after its transformer layers.

## License

The project source code is released under the [Apache License 2.0](LICENSE). The upstream Inkling checkpoint remains subject to its own published license and attribution.
