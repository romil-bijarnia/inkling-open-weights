# Inkling Semantic Map data

The semantic map is a second, relationship-first view of all 200,058 input
tokens. It does not replace the existing PCA galaxy. The galaxy shows the
global shape of the 6,144-dimensional embedding matrix; this map instead tries
to keep tokens with strong measured neighbour relationships near one another.

## Method

The source is the complete `200,058 × 32` neighbour table already derived from
the original 6,144-dimensional BF16 embedding rows. For every token, the input
contains its exact 32 highest-cosine neighbours across the full vocabulary.
Cosine similarity is converted to distance as `1 - cosine`.

UMAP's fuzzy-simplicial-set construction turns those directed neighbour lists
into one weighted undirected graph. A deterministic two-dimensional UMAP run
uses that precomputed graph directly: 32 neighbours, 200 epochs, spectral
initialisation, `min_dist=0.08`, `spread=1.0`, random seed 42, and one optimizer
thread. The final coordinates receive only a translation and a uniform scale,
so their aspect ratio and relative geometry are preserved.

Weighted Leiden community detection runs on the exact same fuzzy graph using
`RBConfigurationVertexPartition`, resolution 1.5, four iterations, and seed 42.
Cluster IDs are then ordered from largest community to smallest. Cluster names
are deliberately data-derived rather than manually asserted: each name uses
readable tokens with high weighted connectivity inside that community, with a
dominant script or token type prefix when that characteristic explains most of
the cluster. Labels are navigation aids, not claims that every member has one
human topic.

## Browser file contract

`data/inkling-semantic-layout.bin` is little-endian Float32. Record `i` belongs
to tokenizer ID `i` and contains exactly `[x, y]`. It therefore has 200,058
records and must be exactly 1,600,464 bytes.

`data/inkling-semantic-clusters.bin` is little-endian Uint32. Element `i`
belongs to tokenizer ID `i` and stores one Leiden cluster ID. It therefore has
200,058 elements and must be exactly 800,232 bytes.

`data/inkling-semantic-manifest.json` describes the method and coordinate
transform, file sizes and hashes, cluster counts, data-derived labels,
representative token IDs, median cluster centroids, cluster bounds, and the
strongest neighbouring clusters. The manifest is the authoritative schema for
browser loading.

## Rebuild and verify

Dependencies are isolated in `.venv-semantic`. Rebuild from the exact neighbour
files with:

```bash
.venv-semantic/bin/python scripts/build_semantic_map.py
```

Hidden graph and UMAP caches can be reused during UI iteration with
`--reuse-cache`; the manifest records whether they were used. Verification uses
only NumPy and runs with:

```bash
.venv-semantic/bin/python scripts/verify_semantic_map.py
```

The generated manifest records the measured runtime, platform, library
versions, final graph size, connected-component count, output sizes, and SHA256
hashes for the two binary browser payloads.

## Measured build

On this Apple-silicon Mac, the uncached fuzzy-graph construction took 2.2
seconds and produced 4,167,789 weighted undirected edges. The complete 200-epoch
UMAP optimization took 116.2 seconds. The final 121-community Leiden run took
80.6 seconds, and metadata generation took 0.6 seconds, for an end-to-end cold
compute time of about 200 seconds. The final packaging run reused the verified
graph and coordinate caches, so its manifest correctly records 81.4 seconds and
marks both cache flags as true.
