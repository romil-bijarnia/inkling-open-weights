# Neural Atlas

Neural Atlas is the live computational-anatomy companion to Inkling Open Weights. It provides a whole-model 3D view, an exact multiresolution atlas of checkpoint weights, and token-by-token functional imaging for open-weight transformers.

The first supported specimen is `EleutherAI/pythia-70m-deduped`, chosen because every parameter and activation can be inspected on ordinary development hardware. The implementation is model-adapter based, so larger GPT-NeoX-style checkpoints can be selected through configuration.

## What is measured

- Every tensor name, shape, dtype, source safetensors file, data range, scalar count, and exact byte address.
- Exact scalar values, raw stored bytes, and raw bit patterns.
- Exact block statistics for multiresolution tensor tiles.
- Per-layer hidden-state norms and changes during inference.
- Attention summaries for the currently generated token.
- Top MLP neuron activations.
- Final logits and an explicitly labelled per-layer logit lens.
- Exact `weight × activation` contributions for a selected MLP output channel.

## What is projected or summarized

The 3D anatomy is a navigational layout rather than a physical arrangement stored in the model. Low-zoom weight cells summarize exact blocks of stored values. Attention displays show selected high-weight edges rather than every matrix entry. Logit-lens results are decoded projections of intermediate residual states, not literal intermediate answers.

## Run it

See [`neural-atlas/README.md`](neural-atlas/README.md) for local, development, Docker, API, and verification instructions.
