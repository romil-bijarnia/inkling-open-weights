# Inkling 3D Open Weights — implementation contract

## Approval and evidence

- Approved direction: the user explicitly requested an interactive 3D visualization of an open model's weights on 2026-07-18, after rejecting generated illustrations.
- Large-screen reference: the user supplied a search screenshot showing a spatial neural-network/tensor visualization. It establishes the 3D inspection intent, not a layout to copy.
- Mobile portrait concept: one selected transformer layer fills the viewport; a 66-layer scrubber and bottom-sheet inspector replace the full desktop rail.
- Mobile landscape concept: the full 66-layer scene is available with two-finger orbit/zoom and explicit reset/focus buttons.
- Purpose: make Inkling's published checkpoint spatially explorable, then reveal a truthful map of real tokenizer rows from `model.llm.embed.weight`, their learned 6,144-dimensional vectors, and their cosine relationships.
- Primary audience: a technically curious viewer who wants to see what “open weights” actually contains.
- Sources: `thinkingmachines/Inkling` at commit `86b4d430ab871652a707666b89203a866888c5e5`, Apache 2.0; the official config, tensor index, all safetensors headers, and byte-range samples from named tensors.
- Update cadence: generated snapshot; refresh only when the data-build script is run.
- Measured layers: tensor names, shapes, dtypes, scalar counts, byte counts, shard locations, router-bias vectors, query-norm vectors, selected matrix slices, tokenizer IDs, exact BF16 embedding rows, vector norms, and original-space cosine similarities.
- Inferred layers: deterministic three-dimensional projection and data-derived embedding clusters. These summarize high-dimensional geometry and are labelled as projections, never as literal coordinates stored in the model.
- Schematic layers: spatial placement and connections between modules. These encode published architecture, not runtime activations.
- Decorative layers: none.
- Truth invariants: architecture layout coordinates are schematic; embedding-map coordinates are a disclosed projection of exact learned rows; all 200,058 tokenizer entries are mapped; neighbour scores come from exhaustive full-vocabulary FP32 cosine comparison in the original 6,144-dimensional space; tokenizer units are not always whole words; never present top router biases as a real token route; never label sampled non-embedding values as the complete 1.9 TB payload; preserve the distinction between 975B advertised total parameters and 952.378B serialized checkpoint scalars.

## Renderer ownership

- Primary renderer: one Three.js/WebGL scene. Depth is meaningful: x is layer order, y is module role, and z separates heads, experts, shared experts, and tensor slices.
- Fallback: a static 2D architecture summary appears only after WebGL initialization or context restoration fails.
- WebGL owns tensor blocks, expert instances, head filaments, the residual spine, real-value surfaces, embedding points, cosine-neighbour edges, camera, and picking geometry.
- HTML owns title, exact values, filters, search, inspector, accessibility mirror, and export controls.
- Render-ready signal: `document.documentElement.dataset.renderReady = "true"` after the first nonblank frame and loaded metadata.
- Lifecycle: one scene, one renderer, capped DPR, `ResizeObserver`, WebGL context-loss notice, and disposal of geometries/materials/listeners on teardown.
- Export: high-resolution PNG from the renderer with the current camera state plus a JSON method/provenance sidecar.
- Desktop bounds: 1280×720 minimum target, responsive to larger displays.
- Mobile portrait: 360–430 px wide, selected-layer focus.
- Mobile landscape: full scene, DPR capped at 1.5.

## Coordinate frames

| Layer | Coordinate system | Transform owner | Alignment check |
| --- | --- | --- | --- |
| Decoder overview | x = layer 0–65, y = module role, z = head/expert index | scene layout | layer labels and picking IDs agree at 0, 29, 65 |
| Expert lattice | x/z = 16×16 expert ID grid, y = learned gate bias | expert layout | IDs 0, 15, 16, 255 map correctly |
| Weight slice | x/y = sample index grid, z = stored numerical value after documented robust scaling | tensor-slice view | first, middle, and last decoded values match JSON |
| Embedding map | x/y/z = deterministic projection of normalized 6,144D token rows | offline data builder | three known token rows and their projection records match binary and JSON |
| Embedding relations | node endpoints = projected tokens; edge value = original-space cosine similarity | offline data builder plus scene edge layer | selected token's stored neighbours recompute from exact rows |
| DOM labels | screen pixels projected from world anchors | overlay projector | resize/orbit label spot checks |

- Perspective projection is used because spatial separation and orbit inspection are part of the task.
- Camera focus uses deterministic named states: overview, layer, experts, tensor slice, embedding overview, and selected token.
- Numerical surface height uses a robust symmetric scale disclosed in the inspector; raw values remain visible.

## Visual encoding ledger

| Data or state | Visual channel | Scale/range | Must not imply |
| --- | --- | --- | --- |
| Layer number | x position | linear 0–65 | training time |
| Module role | y band and hue | fixed semantic bands | performance ranking |
| Tensor scalar count | block volume | log-scaled | exact linear volume comparison |
| Tensor shape | block aspect and inspector | normalized dimensions plus exact text | physical hardware layout |
| Local/global attention | short/long spatial arc | binary | measured attention strength |
| Expert ID | 16×16 lattice position | exact 0–255 | similarity between neighbors |
| Router gate bias | expert height | robust linear scale | actual token routing probability |
| Real sampled weight | slice-surface height and luminance | symmetric robust scale | unsampled values |
| Token embedding | one uniform point | exact row from `model.llm.embed.weight` | a word occurrence or contextual meaning |
| Embedding position | x/y/z | deterministic projection of normalized 6,144D rows | literal stored dimensions or perfect distance preservation |
| Embedding cluster | categorical hue | deterministic data-derived clustering | a human-authored semantic taxonomy |
| Token relation | selected edge plus numeric inspector score | exhaustive full-vocabulary cosine similarity calculated in 6,144D | causation or guaranteed synonymy |
| Raw vector coordinates | signed 64×96 profile in inspector | all 6,144 exact decoded BF16 values | contextual activations |
| Selection | white outline and focus bracket | binary | greater magnitude |

- Color roles: white/graphite context; cobalt attention; violet routed experts; gold shared experts; green vision; coral audio; cyan MTP; white embeddings/norm/output.
- No ambient particles, arbitrary glow, or fake activity animation.
- Essential values visible without hover: 66 layers, 1,552 tensors, 109 files, serialized scalar count, stored bytes, selected layer/tensor, exact sample status.

## Shareability and persistence

- URL state: `view`, `layer`, `tensor`, `expert`, and `token`.
- Local storage: reduced-motion preference and last camera mode only.
- URL state wins over local preferences; invalid state returns to overview.
- Committed selections update history; camera orbit does not.
- Reset clears selection and restores the overview camera.

## Workspace shell

- Shell: single visualization with a collapsible inspector.
- Desktop: compact command bar, central 3D viewport, right inspector.
- Mobile: top title/status, full-height viewport, bottom command bar, inspector bottom sheet.
- Default: overview with layer 29 subtly bracketed as an orientation landmark, not selected.
- Empty-surface drag orbits; empty click clears selection.

## Interaction state machine

| State | Entered by | Visual response | Exit/resume rule |
| --- | --- | --- | --- |
| overview | load/reset | full 66-layer view | select/search a layer or tensor |
| preview | pointer hover | exact mark outline and compact tooltip | pointer leaves |
| selected | click/tap/search | inspector pins and camera focuses | Escape, reset, or another selection |
| tensor slice | inspector action | real numerical sample becomes a 3D surface | Back returns to selected layer |
| embedding overview | Embeddings mode | all 200,058 tokenizer rows, adaptively bright GPU points, and collision-culled labels | search or click selects one token |
| embedding selected | token click/search | projected vector arrow, 32 exhaustive global cosine-neighbour edges, labels, complete vector profile, and inspector | another token traverses the graph; reset returns to embedding overview |

- Mouse: drag orbit, wheel zoom, click select. Touch: one-finger orbit, two-finger zoom/pan, tap select; explicit zoom/reset controls remain available.
- Keyboard: arrow keys step layers, Enter opens details, Escape returns, `/` focuses tensor search.
- Dense picking: Three.js raycasting against instanced layer/expert/tensor objects; the 200,058-point embedding cloud is picked only on committed clicks, with stable point-index-to-token-ID mapping and enlarged screen-space tolerance.
- Reduced motion: camera snaps or uses a very short crossfade; no idle animation.
- No permission-gated device capabilities.

## QA

- Build: `npm run build`
- Data integrity: `python3 scripts/verify_data.py` and `python3 scripts/verify_full_embeddings.py`
- Desktop and mobile screenshots must show a nonblank WebGL canvas, readable overlays, and the same selected entity.
- Smoke tests: orbit, zoom, reset, tensor search, token search, layer stepping, expert selection, tensor slice, embedding selection, neighbour traversal, vector-profile decode, export, URL restore, reduced motion, and WebGL fallback.
- Pixel check: exported canvas has nonblack pixels after render-ready.
- Coordinate checks: layers 0/29/65, experts 0/255, three sampled non-embedding weight values, all 200,058 embedding token IDs, exact 6,144-value row lengths, and exhaustive cosine recomputation for deterministic probe tokens.
