import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./styles.css";

const COLORS = {
  attention: 0x7aa2ff,
  routed_experts: 0xaa78ff,
  shared_experts: 0xf0c56a,
  router: 0x82c8e8,
  dense_mlp: 0x9b83d9,
  normalization: 0xd9e2e7,
  embedding: 0xf0f4f6,
  output: 0xf0f4f6,
  vision: 0x66d9a8,
  audio: 0xff816c,
  next_token_prediction: 0x5ed8df,
  other: 0x7e8b94,
};

const CATEGORY_LABELS = {
  attention: "Attention",
  routed_experts: "Routed experts",
  shared_experts: "Shared experts",
  router: "Router",
  dense_mlp: "Dense MLP",
  normalization: "Normalization",
  embedding: "Embedding",
  output: "Output",
  vision: "Vision adapter",
  audio: "Audio adapter",
  next_token_prediction: "Multi-token prediction",
  other: "Other",
};

const Y_BANDS = {
  attention: 3.4,
  routed_experts: -3.65,
  shared_experts: -5.55,
  router: -1.55,
  dense_mlp: -3.65,
  normalization: 0.65,
  embedding: 0,
  output: 0,
  vision: 5.4,
  audio: 3.15,
  next_token_prediction: -8.4,
  other: 0,
};

const Z_SPANS = {
  attention: 3.5,
  routed_experts: 3.4,
  shared_experts: 1.1,
  router: 0.7,
  dense_mlp: 2.6,
  normalization: 1,
  embedding: 1.2,
  output: 1.2,
  vision: 2.2,
  audio: 2.2,
  next_token_prediction: 2.4,
  other: 1.6,
};

const DEFAULT_LAYER = 29;
const DEFAULT_SAMPLE = "model.llm.layers.29.attn.wq_du.weight";
const EMBEDDING_MANIFEST_URL = "/inkling-embedding-full-manifest.json";
const EMBEDDING_VOCAB_URL = "/inkling-embedding-vocab.json";
const EMBEDDING_LAYOUT_URL = "/inkling-embedding-layout.bin";
const EMBEDDING_NEIGHBOR_IDS_URL = "/inkling-embedding-neighbor-ids.bin";
const EMBEDDING_NEIGHBOR_COSINES_URL = "/inkling-embedding-neighbor-cosines.bin";
const EMBEDDING_VECTOR_URL = "/inkling-embedding-vectors-full.bin";
const SEMANTIC_MANIFEST_URL = "/inkling-semantic-manifest.json";
const SEMANTIC_LAYOUT_URL = "/inkling-semantic-layout.bin";
const SEMANTIC_CLUSTERS_URL = "/inkling-semantic-clusters.bin";
const EMBEDDING_DIMENSIONS = 6144;
const EMBEDDING_ROW_BYTES = EMBEDDING_DIMENSIONS * 2;
const EXPECTED_TOKEN_COUNT = 200_058;
const DEFAULT_NEIGHBORS_PER_TOKEN = 32;
const LOCAL_SETUP_COMMANDS = `git clone https://github.com/romil-bijarnia/inkling-open-weight-atlas.git
cd inkling-open-weight-atlas
npm ci
npm run dev

# Optional: exact 6,144-value token views
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm run download:embeddings:full`;
const EMBEDDING_CLUSTER_COLORS = [
  0x6f90d9, 0x8f75c9, 0x5fa9b8, 0x7f9e73, 0xb18a68, 0x9d708c,
  0x668f9f, 0xa47b60, 0x7788b8, 0x6e9c91, 0x927ab0, 0x8d9270,
];
const EMBEDDING_LABEL_ANCHOR_RANK = new Map(`
person people human man woman boy girl child family mother father sister brother friend love hate happy sad joy fear anger calm hope trust truth
king queen prince princess leader worker teacher doctor nurse artist writer musician scientist dog cat horse lion tiger bear wolf fox bird fish whale
tree flower forest river ocean sea mountain valley desert rain snow wind fire earth water air sun moon star planet space universe world nature life death
body mind heart brain eye hand red blue green yellow orange purple black white colour light dark hot cold big small old young one two three four five ten
today tomorrow yesterday morning evening night time year month week day future past history home house room city village country nation street road school
university hospital market bank australia melbourne sydney india china japan france germany italy spain london paris language word sentence story book music
song film movie art game sport food coffee tea bread science physics chemistry biology mathematics number equation energy matter atom cell gene computer
machine software hardware data code program function class model network algorithm artificial intelligence learning knowledge reason memory attention token
vector embedding internet web cloud server database python javascript java html json linux apple microsoft work money business company government law politics
society culture religion health education car train plane ship phone camera robot tool question answer idea problem solution cause effect change create build
make use know think feel run walk speak hear see read write learn teach help choose give take open close fast slow strong weak high low near far left right
inside outside before after begin end freedom peace war power justice equality beauty danger safety success failure possible impossible
`.trim().split(/\s+/).map((word, index) => [word, index]));
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const dom = {
  canvas: document.querySelector("#weight-space"),
  viewport: document.querySelector(".viewport-shell"),
  loading: document.querySelector("#loading-panel"),
  fallback: document.querySelector("#webgl-fallback"),
  fallbackSummary: document.querySelector("#fallback-summary"),
  sourceStatus: document.querySelector("#source-status"),
  sourceLink: document.querySelector("#source-link"),
  commitLabel: document.querySelector("#commit-label"),
  searchLabel: document.querySelector("#search-label"),
  search: document.querySelector("#tensor-search"),
  searchResults: document.querySelector("#search-results"),
  tooltip: document.querySelector("#tooltip"),
  sceneKey: document.querySelector("#scene-key"),
  axisGuide: document.querySelector("#axis-guide"),
  layerControl: document.querySelector(".layer-control"),
  viewInstructions: document.querySelector("#view-instructions"),
  layerSlider: document.querySelector("#layer-slider"),
  layerLabel: document.querySelector("#layer-label"),
  layerKind: document.querySelector("#layer-kind"),
  embeddingControl: document.querySelector("#embedding-control"),
  embeddingCount: document.querySelector("#embedding-count"),
  embeddingViewButtons: [...document.querySelectorAll("[data-embedding-view]")],
  embeddingNeighborLens: document.querySelector("#embedding-neighbor-lens"),
  embeddingMeaningMap: document.querySelector("#embedding-meaning-map"),
  embeddingLabelLayer: document.querySelector("#embedding-label-layer"),
  labelDensityButtons: [...document.querySelectorAll("[data-label-density]")],
  inspector: document.querySelector("#inspector"),
  inspectorKicker: document.querySelector("#inspector-kicker"),
  inspectorTitle: document.querySelector("#inspector-title"),
  inspectorBody: document.querySelector("#inspector-body"),
  accessibleSummary: document.querySelector("#accessible-summary"),
  statParams: document.querySelector("#stat-params"),
  statSize: document.querySelector("#stat-size"),
  statTensors: document.querySelector("#stat-tensors"),
  statFiles: document.querySelector("#stat-files"),
  resetView: document.querySelector("#reset-view"),
  exportView: document.querySelector("#export-view"),
  projectMenuWrap: document.querySelector(".project-menu-wrap"),
  projectMenuButton: document.querySelector("#project-menu-button"),
  projectMenu: document.querySelector("#project-menu"),
  projectMenuClose: document.querySelector("#project-menu-close"),
  copyLocalSetup: document.querySelector("#copy-local-setup"),
  menuToggleInspector: document.querySelector("#menu-toggle-inspector"),
  closeInspector: document.querySelector("#close-inspector"),
  modeButtons: [...document.querySelectorAll(".mode-button")],
};

const state = {
  data: null,
  mode: "overview",
  selectedLayer: DEFAULT_LAYER,
  selectedTensor: null,
  selectedExpert: null,
  selectedValue: null,
  selectedToken: null,
  pendingTokenId: null,
  embeddingData: null,
  embeddingView: "all",
  embeddingLabelDensity: "auto",
  activeValueSample: null,
  pendingExpertId: null,
  hovered: null,
  inspectorOpen: true,
  pointerDown: null,
  sceneReady: false,
  renderActive: true,
};

let renderer;
let scene;
let camera;
let controls;
let raycaster;
let pointer;
let architectureGroup;
let tensorGroup;
let tensorMesh;
let tensorLayouts = [];
let layerHitMesh;
let expertDetailGroup;
let expertMesh;
let expertInstanceData = [];
let valuesGroup;
let valuesMesh;
let valueInstanceData = [];
let embeddingsGroup;
let embeddingPoints;
let embeddingFocusGroup;
let embeddingVisualPositions = new Float32Array(0);
let embeddingNorms = new Float32Array(0);
let embeddingLabels = [];
let embeddingLabelCandidateIds = [];
let embeddingAnchorCandidateCount = 0;
let embeddingOverviewSphere = null;
let embeddingLoadPromise = null;
let embeddingLabelRefreshTimer = null;
let embeddingSelectionGeneration = 0;
let embeddingSearchTimer = null;
let embeddingActiveNeighbors = [];
let semanticMapData = null;
let semanticMapLoadPromise = null;
let semanticMapCanvas = null;
let semanticMapContext = null;
let semanticMapFrame = null;
let semanticMapPointer = null;
const semanticMapTransform = { scale: 1, offsetX: 0, offsetY: 0 };
const embeddingVectorCache = new Map();
const embeddingNeighborCache = new Map();
let selectionBox;
let hoverBox;
let cameraTransition = null;
let resizeObserver;

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 2) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(value);
}

function formatCount(value) {
  if (value >= 1e12) return `${(value / 1e12).toFixed(3)}T`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(3)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
  return formatNumber(value, 0);
}

function formatBytes(value) {
  if (value >= 1e12) return `${(value / 1e12).toFixed(3)} TB`;
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} GB`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)} MB`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(2)} KB`;
  return `${value} B`;
}

function shapeText(shape) {
  return `[${shape.map((value) => formatNumber(value, 0)).join(" × ")}]`;
}

function hashString(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function unitHash(value, salt = 0) {
  const hash = hashString(`${salt}:${value}`);
  return hash / 0xffffffff;
}

function layerX(layer) {
  return (layer - 32.5) * 1.66;
}

function mtpLayerFromName(name) {
  const match = name.match(/model\.mtp\.layers\.(\d+)\./);
  return match ? Number(match[1]) : null;
}

function colorFor(category) {
  return COLORS[category] ?? COLORS.other;
}

function visibleToken(token) {
  if (!token) return "unknown token";
  let value = String(token.display ?? token.raw ?? token.id ?? "");
  if (value.startsWith("Ġ")) value = `␠${value.slice(1)}`;
  value = value.replace(/^( +)/, (spaces) => "␠".repeat(spaces.length));
  return value || "∅";
}

function rawToken(token) {
  return String(token?.raw ?? token?.display ?? token?.id ?? "");
}

function tokenIdentifier(token) {
  return String(token?.id ?? token?.index ?? "");
}

function clusterIdentifier(cluster, fallback) {
  return String(cluster?.id ?? cluster?.index ?? cluster?.cluster ?? fallback);
}

function clusterColor(clusterId) {
  const clusters = state.embeddingData?.clusters ?? [];
  const clusterIndex = clusters.findIndex((cluster, index) => clusterIdentifier(cluster, index) === String(clusterId));
  const record = clusterIndex >= 0 ? clusters[clusterIndex] : null;
  if (typeof record?.color === "string" && /^#?[0-9a-f]{6}$/i.test(record.color)) {
    return Number.parseInt(record.color.replace("#", ""), 16);
  }
  const numeric = clusterIndex >= 0 ? clusterIndex : Math.abs(hashString(String(clusterId)));
  return EMBEDDING_CLUSTER_COLORS[numeric % EMBEDDING_CLUSTER_COLORS.length];
}

function projectionVarianceText() {
  const projection = state.embeddingData?.projection ?? {};
  const values = projection.explainedVariance ?? projection.explainedVarianceRatio ?? projection.explainedVarianceRatios ?? projection.varianceRatio ?? projection.variance ?? [];
  if (!Array.isArray(values) || !values.length) return "reported in source map";
  return values.slice(0, 3).map((value) => `${(Number(value) * 100).toFixed(2)}%`).join(" · ");
}

function embeddingTokenFromReference(reference) {
  if (reference === null || reference === undefined) return null;
  if (typeof reference === "object" && (reference.id !== undefined || reference.index !== undefined)) return reference;
  const numeric = Number(String(reference).replace(/^token\s*#?/i, ""));
  if (Number.isInteger(numeric) && numeric >= 0 && numeric < (state.embeddingData?.tokens?.length ?? 0)) {
    const direct = state.embeddingData.tokens[numeric];
    if (Number(direct?.id ?? numeric) === numeric) return direct;
  }
  return null;
}

function neighborToken(record) {
  return embeddingTokenFromReference(record?.id ?? record?.index);
}

function embeddingManifestValue(...keys) {
  for (const key of keys) {
    const value = state.embeddingData?.manifest?.[key] ?? state.embeddingData?.[key];
    if (value !== undefined && value !== null) return value;
  }
  return undefined;
}

function setMaterialOpacity(object, opacity) {
  object.traverse((child) => {
    if (child.material) {
      child.material.transparent = opacity < 1;
      child.material.opacity = opacity;
      child.material.needsUpdate = true;
    }
  });
}

function categoryTotalsHTML(categoryParameters) {
  return Object.entries(categoryParameters)
    .sort((a, b) => b[1] - a[1])
    .map(
      ([category, parameters]) => `
        <div class="detail-row">
          <dt>${escapeHTML(CATEGORY_LABELS[category] ?? category)}</dt>
          <dd>${formatCount(parameters)}</dd>
        </div>`,
    )
    .join("");
}

function hasWebGL() {
  try {
    const probe = document.createElement("canvas");
    return Boolean(probe.getContext("webgl2") || probe.getContext("webgl"));
  } catch {
    return false;
  }
}

function initThree() {
  if (!hasWebGL()) throw new Error("WebGL is not available");

  renderer = new THREE.WebGLRenderer({
    canvas: dom.canvas,
    antialias: true,
    alpha: false,
    preserveDrawingBuffer: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(0x030507, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, window.innerWidth < 720 ? 1.5 : 2));

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x030507, 0.0065);
  camera = new THREE.PerspectiveCamera(36, 1, 0.1, 900);
  camera.position.set(72, 39, 78);

  controls = new OrbitControls(camera, dom.canvas);
  controls.enableDamping = !reducedMotion;
  controls.dampingFactor = 0.075;
  controls.minDistance = 7;
  controls.maxDistance = 180;
  controls.screenSpacePanning = true;
  controls.target.set(0, -3.15, 0);
  controls.update();
  controls.addEventListener("change", () => {
    if (state.mode === "embeddings" && !state.selectedToken) scheduleEmbeddingLabelRefresh(false);
  });

  raycaster = new THREE.Raycaster();
  raycaster.params.Line.threshold = 0.18;
  raycaster.params.Points.threshold = 0.7;
  pointer = new THREE.Vector2(2, 2);

  const ambient = new THREE.AmbientLight(0xb9d1de, 1.15);
  const key = new THREE.DirectionalLight(0xe7f4ff, 2.2);
  key.position.set(25, 32, 22);
  const rim = new THREE.DirectionalLight(0x768fff, 1.3);
  rim.position.set(-30, 5, -24);
  scene.add(ambient, key, rim);

  architectureGroup = new THREE.Group();
  architectureGroup.name = "architecture";
  tensorGroup = new THREE.Group();
  tensorGroup.name = "tensors";
  expertDetailGroup = new THREE.Group();
  expertDetailGroup.name = "expert-detail";
  valuesGroup = new THREE.Group();
  valuesGroup.name = "value-samples";
  embeddingsGroup = new THREE.Group();
  embeddingsGroup.name = "embedding-map";
  embeddingFocusGroup = new THREE.Group();
  embeddingFocusGroup.name = "embedding-focus";
  embeddingsGroup.add(embeddingFocusGroup);
  scene.add(architectureGroup, tensorGroup, expertDetailGroup, valuesGroup, embeddingsGroup);

  selectionBox = createWireBox(0xffffff, 0.95);
  selectionBox.visible = false;
  hoverBox = createWireBox(0x9fd6f2, 0.72);
  hoverBox.visible = false;
  scene.add(selectionBox, hoverBox);

  dom.canvas.addEventListener("webglcontextlost", onContextLost, false);
  dom.canvas.addEventListener("webglcontextrestored", () => window.location.reload(), false);
  resizeObserver = new ResizeObserver(resizeRenderer);
  resizeObserver.observe(dom.viewport);
  resizeRenderer();
}

function createWireBox(color, opacity) {
  const geometry = new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1));
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthTest: false });
  const box = new THREE.LineSegments(geometry, material);
  box.renderOrder = 20;
  return box;
}

function onContextLost(event) {
  event.preventDefault();
  state.renderActive = false;
  showFallback("The WebGL context was lost. Reload the page to rebuild the 3D atlas.");
}

function resizeRenderer() {
  if (!renderer) return;
  const width = Math.max(1, dom.viewport.clientWidth);
  const height = Math.max(1, dom.viewport.clientHeight);
  const pixelRatio = Math.min(window.devicePixelRatio || 1, width < 720 ? 1.5 : 2);
  renderer.setPixelRatio(pixelRatio);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  if (embeddingPoints?.material?.uniforms?.uPixelRatio) {
    embeddingPoints.material.uniforms.uPixelRatio.value = pixelRatio;
    embeddingPoints.material.uniforms.uPointScale.value = width < 720 ? 3.2 : 2.75;
  }
  embeddingFocusGroup?.traverse?.((child) => {
    if (child.material?.uniforms?.uPixelRatio) child.material.uniforms.uPixelRatio.value = pixelRatio;
  });
  if (state.mode === "embeddings" && !state.selectedToken) scheduleEmbeddingLabelRefresh(false);
  resizeSemanticMap();
}

function buildScene(data) {
  buildReferenceGrid();
  buildArchitecture(data);
  buildTensorMesh(data);
  buildLayerHitTargets(data);
  buildExpertDetail(data, state.selectedLayer);
  buildValueView(data, sampleForCurrentSelection());
  applyModeVisibility();
}

function buildReferenceGrid() {
  const grid = new THREE.GridHelper(130, 65, 0x23313a, 0x10181d);
  grid.position.y = -8.9;
  grid.material.transparent = true;
  grid.material.opacity = 0.36;
  architectureGroup.add(grid);

  const axes = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-58, 0, 0),
    new THREE.Vector3(59, 0, 0),
  ]);
  const axisLine = new THREE.Line(axes, new THREE.LineBasicMaterial({ color: 0xd3e2e9, transparent: true, opacity: 0.7 }));
  architectureGroup.add(axisLine);
}

function buildArchitecture(data) {
  const framePositions = [];
  const frameColors = [];
  const localColor = new THREE.Color(0x33434d);
  const globalColor = new THREE.Color(0x8bb5d1);
  const residualPositions = [];
  const attentionPositions = [];
  const expertPositions = [];
  const sharedPositions = [];

  for (const layer of data.layers) {
    const x = layerX(layer.id);
    addBoxEdges(framePositions, frameColors, x, -0.2, 0, 0.74, 12.5, 8.2, layer.attention === "global" ? globalColor : localColor);
    residualPositions.push(x - 0.66, 0, 0, x + 0.66, 0, 0);

    const attentionHeight = layer.attention === "global" ? 5.1 : 3.5;
    const attentionSpan = layer.attention === "global" ? 3.8 : 2.6;
    for (let head = 0; head < data.architecture.attentionHeads; head += 1) {
      const z = ((head / (data.architecture.attentionHeads - 1)) * 2 - 1) * attentionSpan;
      attentionPositions.push(x - 0.58, 0, 0, x, attentionHeight, z);
      attentionPositions.push(x, attentionHeight, z, x + 0.58, 0, 0);
    }

    if (layer.id >= 2) {
      const router = [x, -1.45, 0];
      for (let expert = 0; expert < data.architecture.routedExperts; expert += 1) {
        const row = Math.floor(expert / 16);
        const column = expert % 16;
        const y = -2.45 - row * 0.23;
        const z = (column - 7.5) * 0.43;
        expertPositions.push(...router, x, y, z);
        expertPositions.push(x, y, z, x + 0.58, 0, 0);
      }
      for (let shared = 0; shared < 2; shared += 1) {
        const z = shared === 0 ? -0.65 : 0.65;
        sharedPositions.push(...router, x, -6.15, z);
        sharedPositions.push(x, -6.15, z, x + 0.58, 0, 0);
      }
    } else {
      for (let strand = 0; strand < 64; strand += 1) {
        const z = ((strand / 63) * 2 - 1) * 2.4;
        expertPositions.push(x - 0.58, 0, 0, x, -4.2, z);
        expertPositions.push(x, -4.2, z, x + 0.58, 0, 0);
      }
    }
  }

  architectureGroup.add(
    lineSegments(framePositions, 0xffffff, 0.38, frameColors),
    lineSegments(residualPositions, 0xd9e4ea, 0.86),
    lineSegments(attentionPositions, COLORS.attention, 0.16),
    lineSegments(expertPositions, COLORS.routed_experts, 0.045),
    lineSegments(sharedPositions, COLORS.shared_experts, 0.34),
  );

  const inputLines = [];
  const inputY = [5.4, 3.15, 0];
  for (const y of inputY) {
    for (let index = 0; index < 16; index += 1) {
      const z = (index - 7.5) * 0.24;
      inputLines.push(-63, y, z, layerX(0) - 0.8, 0, z * 0.18);
    }
  }
  architectureGroup.add(lineSegments(inputLines, 0xb9d6e5, 0.18));

  const mtpLines = [];
  for (let index = 0; index < 8; index += 1) {
    const x = -14 + index * 4;
    mtpLines.push(x, -8.4, -2.6, x, -8.4, 2.6);
  }
  architectureGroup.add(lineSegments(mtpLines, COLORS.next_token_prediction, 0.36));
}

function addBoxEdges(target, colorTarget, x, y, z, sx, sy, sz, color) {
  const corners = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ].map(([cx, cy, cz]) => [x + (cx * sx) / 2, y + (cy * sy) / 2, z + (cz * sz) / 2]);
  const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
  for (const [start, end] of edges) {
    target.push(...corners[start], ...corners[end]);
    colorTarget.push(color.r, color.g, color.b, color.r, color.g, color.b);
  }
}

function lineSegments(positions, color, opacity, colors = null) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  if (colors?.length) geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    color: colors?.length ? 0xffffff : color,
    vertexColors: Boolean(colors?.length),
    transparent: true,
    opacity,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  return new THREE.LineSegments(geometry, material);
}

function tensorLayout(tensor) {
  const mass = Math.max(0, Math.min(1, (Math.log10(Math.max(1, tensor.parameters)) - 2) / 8));
  const jitterX = (unitHash(tensor.name, 1) - 0.5) * 0.44;
  const span = Z_SPANS[tensor.category] ?? Z_SPANS.other;
  let x;
  let y = Y_BANDS[tensor.category] ?? 0;
  let z = (unitHash(tensor.name, 2) * 2 - 1) * span;

  if (tensor.layer !== null) {
    x = layerX(tensor.layer) + jitterX;
  } else {
    const mtpLayer = mtpLayerFromName(tensor.name);
    if (mtpLayer !== null) {
      x = -14 + mtpLayer * 4 + jitterX;
      y = Y_BANDS.next_token_prediction + (unitHash(tensor.name, 3) - 0.5) * 1.1;
    } else if (tensor.category === "vision") {
      x = -61 + unitHash(tensor.name, 4) * 3;
    } else if (tensor.category === "audio") {
      x = -61 + unitHash(tensor.name, 4) * 3;
    } else if (tensor.category === "embedding") {
      x = -61 + unitHash(tensor.name, 4) * 3;
    } else if (tensor.name.includes("unembed") || tensor.name.includes("lm_head") || tensor.category === "output") {
      x = 58.5 + unitHash(tensor.name, 4) * 2.5;
    } else {
      x = 57.5 + unitHash(tensor.name, 4) * 2.2;
    }
  }

  const scale = {
    x: 0.1 + mass * 0.28,
    y: 0.1 + mass * 0.58,
    z: 0.1 + mass * 0.82,
  };
  if (tensor.category === "routed_experts") {
    scale.y *= 1.24;
    scale.z *= 1.55;
  }
  if (tensor.category === "shared_experts") scale.z *= 1.18;
  return { position: new THREE.Vector3(x, y, z), scale: new THREE.Vector3(scale.x, scale.y, scale.z) };
}

function buildTensorMesh(data) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshStandardMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.83,
    roughness: 0.48,
    metalness: 0.12,
  });
  tensorMesh = new THREE.InstancedMesh(geometry, material, data.tensors.length);
  tensorMesh.instanceMatrix.setUsage(THREE.StaticDrawUsage);
  tensorMesh.userData.kind = "tensor";
  tensorLayouts = [];
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const color = new THREE.Color();

  data.tensors.forEach((tensor, index) => {
    const layout = tensorLayout(tensor);
    tensorLayouts.push(layout);
    matrix.compose(layout.position, quaternion, layout.scale);
    tensorMesh.setMatrixAt(index, matrix);
    color.setHex(colorFor(tensor.category));
    const lightnessShift = (unitHash(tensor.name, 6) - 0.5) * 0.12;
    color.offsetHSL(0, 0, lightnessShift);
    tensorMesh.setColorAt(index, color);
  });
  tensorMesh.instanceMatrix.needsUpdate = true;
  tensorMesh.instanceColor.needsUpdate = true;
  tensorMesh.computeBoundingBox();
  tensorMesh.computeBoundingSphere();
  tensorGroup.add(tensorMesh);
}

function buildLayerHitTargets(data) {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
  layerHitMesh = new THREE.InstancedMesh(geometry, material, data.layers.length);
  layerHitMesh.userData.kind = "layer";
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const scale = new THREE.Vector3(0.95, 12.7, 8.4);
  data.layers.forEach((layer, index) => {
    matrix.compose(new THREE.Vector3(layerX(layer.id), -0.2, 0), quaternion, scale);
    layerHitMesh.setMatrixAt(index, matrix);
  });
  layerHitMesh.instanceMatrix.needsUpdate = true;
  layerHitMesh.computeBoundingBox();
  layerHitMesh.computeBoundingSphere();
  architectureGroup.add(layerHitMesh);
}

function buildExpertDetail(data, layerId) {
  while (expertDetailGroup.children.length) disposeObject(expertDetailGroup.children.pop());
  expertInstanceData = [];
  expertMesh = null;
  const layer = data.layers[layerId];
  if (!layer || layerId < 2 || !layer.gateBias?.values) return;

  const biases = layer.gateBias.values;
  const mean = biases.reduce((sum, value) => sum + value, 0) / biases.length;
  const deviations = biases.map((value) => value - mean);
  const maxAbs = Math.max(...deviations.map(Math.abs), 1e-9);
  const ranked = [...biases.keys()].sort((a, b) => biases[b] - biases[a]);
  const topSix = new Set(ranked.slice(0, data.architecture.expertsPerToken));

  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
  expertMesh = new THREE.InstancedMesh(geometry, material, biases.length);
  expertMesh.userData.kind = "expert";
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const color = new THREE.Color();
  biases.forEach((bias, expert) => {
    const row = Math.floor(expert / 16);
    const column = expert % 16;
    const extrusion = (bias - mean) / maxAbs;
    const depth = 0.16 + Math.abs(extrusion) * 1.15;
    const position = new THREE.Vector3(layerX(layerId) + extrusion * 0.68, -5.85 + row * 0.39, -3.05 + column * 0.405);
    const scale = new THREE.Vector3(depth, 0.25, 0.25);
    matrix.compose(position, quaternion, scale);
    expertMesh.setMatrixAt(expert, matrix);
    color.setHex(topSix.has(expert) ? 0x73d7f5 : COLORS.routed_experts);
    expertMesh.setColorAt(expert, color);
    expertInstanceData.push({ expert, bias, position, scale, topBias: topSix.has(expert) });
  });
  expertMesh.instanceMatrix.needsUpdate = true;
  expertMesh.instanceColor.needsUpdate = true;
  expertMesh.computeBoundingBox();
  expertMesh.computeBoundingSphere();
  expertDetailGroup.add(expertMesh);

  const sharedGeometry = new THREE.BoxGeometry(1, 1, 1);
  const sharedMaterial = new THREE.MeshStandardMaterial({ color: COLORS.shared_experts, roughness: 0.38 });
  for (let index = 0; index < data.architecture.sharedExperts; index += 1) {
    const shared = new THREE.Mesh(sharedGeometry, sharedMaterial.clone());
    shared.scale.set(1.15, 0.32, 0.5);
    shared.position.set(layerX(layerId), 1.0 + index * 0.55, -3.75);
    shared.userData.kind = "shared-expert";
    shared.userData.expert = index;
    expertDetailGroup.add(shared);
  }

  const frame = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1.7, 6.75, 6.75)),
    new THREE.LineBasicMaterial({ color: 0x7893a3, transparent: true, opacity: 0.45 }),
  );
  frame.position.set(layerX(layerId), -2.9, 0);
  expertDetailGroup.add(frame);
}

function sampleForCurrentSelection() {
  if (!state.data) return null;
  if (state.activeValueSample) return state.activeValueSample;
  const selectedName = state.selectedTensor?.name;
  if (selectedName && state.data.matrixSamples[selectedName]) return state.data.matrixSamples[selectedName];
  return state.data.matrixSamples[DEFAULT_SAMPLE] ?? Object.values(state.data.matrixSamples)[0];
}

function buildValueView(data, sample) {
  while (valuesGroup.children.length) disposeObject(valuesGroup.children.pop());
  valueInstanceData = [];
  valuesMesh = null;
  if (!sample) return;

  const flatValues = sample.values.flat();
  const absSorted = flatValues.map(Math.abs).sort((a, b) => a - b);
  const robustMax = absSorted[Math.min(absSorted.length - 1, Math.floor(absSorted.length * 0.99))] || 1;
  const rows = sample.values.length;
  const columns = sample.values[0]?.length ?? 0;
  const spacing = Math.min(0.34, 11.2 / Math.max(rows - 1, columns - 1, 1));
  const barWidth = Math.min(0.24, spacing * 0.72);
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff });
  valuesMesh = new THREE.InstancedMesh(geometry, material, rows * columns);
  valuesMesh.userData.kind = "value";
  const matrix = new THREE.Matrix4();
  const quaternion = new THREE.Quaternion();
  const color = new THREE.Color();
  let instance = 0;
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const value = sample.values[row][column];
      const normalized = THREE.MathUtils.clamp(value / robustMax, -1, 1);
      const height = 0.06 + Math.abs(normalized) * 4.1;
      const position = new THREE.Vector3(
        (column - (columns - 1) / 2) * spacing,
        Math.sign(normalized || 1) * height * 0.5,
        (row - (rows - 1) / 2) * spacing,
      );
      const scale = new THREE.Vector3(barWidth, height, barWidth);
      matrix.compose(position, quaternion, scale);
      valuesMesh.setMatrixAt(instance, matrix);
      if (normalized < 0) color.setRGB(0.25 + (1 - Math.abs(normalized)) * 0.22, 0.48 + (1 - Math.abs(normalized)) * 0.22, 1);
      else color.setRGB(1, 0.34 + (1 - normalized) * 0.38, 0.24 + (1 - normalized) * 0.5);
      valuesMesh.setColorAt(instance, color);
      valueInstanceData.push({
        instance,
        value,
        row: sample.rowIndices[row],
        column: sample.columnIndices[column],
        leadingIndices: sample.leadingIndices,
        storedIndex: sample.kind === "embedding-vector"
          ? [sample.vectorTokenId, row * columns + column]
          : [...sample.leadingIndices, sample.rowIndices[row], sample.columnIndices[column]],
        position,
        scale,
      });
      instance += 1;
    }
  }
  valuesMesh.instanceMatrix.needsUpdate = true;
  valuesMesh.instanceColor.needsUpdate = true;
  valuesMesh.computeBoundingBox();
  valuesMesh.computeBoundingSphere();
  valuesGroup.add(valuesMesh);

  const zeroPlane = new THREE.GridHelper(12, 32, 0x6d7c85, 0x182229);
  zeroPlane.material.transparent = true;
  zeroPlane.material.opacity = 0.42;
  valuesGroup.add(zeroPlane);

  const axisGeometry = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-5.6, 0, 5.8), new THREE.Vector3(5.6, 0, 5.8),
    new THREE.Vector3(-5.8, -4.4, 5.8), new THREE.Vector3(-5.8, 4.4, 5.8),
    new THREE.Vector3(-5.8, 0, -5.6), new THREE.Vector3(-5.8, 0, 5.8),
  ]);
  valuesGroup.add(new THREE.LineSegments(axisGeometry, new THREE.LineBasicMaterial({ color: 0xa9bac4, transparent: true, opacity: 0.65 })));
}

function createEmbeddingPointTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(32, 32, 2, 32, 32, 30);
  gradient.addColorStop(0, "rgba(255,255,255,1)");
  gradient.addColorStop(0.58, "rgba(255,255,255,0.92)");
  gradient.addColorStop(1, "rgba(255,255,255,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 64, 64);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function embeddingArrayPosition(token) {
  const tokenId = Number(token?.id ?? token?.index);
  if (Number.isInteger(tokenId) && tokenId >= 0 && tokenId < (state.embeddingData?.tokens?.length ?? 0)) return tokenId;
  return state.embeddingData?.tokens?.indexOf(token) ?? -1;
}

function embeddingVisualPosition(token, target = new THREE.Vector3()) {
  const index = embeddingArrayPosition(token);
  const offset = index * 3;
  if (index < 0 || offset + 2 >= embeddingVisualPositions.length) return null;
  return target.set(embeddingVisualPositions[offset], embeddingVisualPositions[offset + 1], embeddingVisualPositions[offset + 2]);
}

function embeddingTokenType(record) {
  return String(record?.type ?? record?.kind ?? record?.category ?? "token");
}

function normalizeVocabulary(payload, expectedCount) {
  let records = Array.isArray(payload) ? payload : payload?.tokens ?? payload?.vocab ?? payload?.entries;
  if (!Array.isArray(records) && Array.isArray(payload?.display)) {
    records = payload.display.map((display, id) => ({
      id,
      display,
      raw: payload.raw?.[id] ?? display,
      type: payload.type?.[id] ?? "token",
      script: payload.script?.[id] ?? "None",
      leadingSpace: Boolean(payload.leadingSpace?.[id]),
    }));
  }
  if (!Array.isArray(records)) throw new Error("Full embedding vocabulary has no token array");
  if (records.length !== expectedCount) {
    throw new Error(`Full embedding vocabulary has ${formatNumber(records.length, 0)} rows; expected ${formatNumber(expectedCount, 0)}`);
  }
  return records.map((record, arrayIndex) => {
    const token = typeof record === "string" ? { display: record, raw: record } : record;
    const id = Number(token.id ?? token.tokenId ?? token.index ?? arrayIndex);
    const display = token.display ?? token.text ?? token.decoded ?? token.token ?? token.raw ?? String(id);
    const raw = token.raw ?? token.piece ?? token.tokenizerPiece ?? display;
    return {
      ...token,
      id,
      index: id,
      display,
      raw,
      type: embeddingTokenType(token),
      leadingSpace: token.leadingSpace ?? String(display).startsWith(" "),
    };
  });
}

function manifestTokenCount(manifest) {
  return Number(manifest.tokenCount ?? manifest.tokenizerEntries ?? manifest.vocabSize ?? manifest.tokens ?? EXPECTED_TOKEN_COUNT);
}

function manifestNeighborCount(manifest) {
  return Number(manifest.neighborsPerToken ?? manifest.neighbourCount ?? manifest.neighbors?.count ?? manifest.neighbours?.count ?? DEFAULT_NEIGHBORS_PER_TOKEN);
}

async function fetchRequired(url, type = "json") {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} request failed: ${response.status}`);
  return type === "arrayBuffer" ? response.arrayBuffer() : response.json();
}

async function loadEmbeddingData() {
  if (state.embeddingData) return state.embeddingData;
  if (embeddingLoadPromise) return embeddingLoadPromise;
  embeddingLoadPromise = (async () => {
    const [manifest, vocabPayload, layoutBuffer] = await Promise.all([
      fetchRequired(EMBEDDING_MANIFEST_URL),
      fetchRequired(EMBEDDING_VOCAB_URL),
      fetchRequired(EMBEDDING_LAYOUT_URL, "arrayBuffer"),
    ]);
    const tokenCount = manifestTokenCount(manifest);
    const tokens = normalizeVocabulary(vocabPayload, tokenCount);
    const layoutStride = Number(manifest.layoutStrideFloats ?? manifest.layout?.strideFloats ?? manifest.layout?.stride ?? 4);
    if (layoutStride < 4) throw new Error(`Embedding layout stride ${layoutStride} does not include xyz and norm`);
    if (layoutBuffer.byteLength !== tokenCount * layoutStride * 4) {
      throw new Error(`Embedding layout returned ${formatNumber(layoutBuffer.byteLength, 0)} bytes; expected ${formatNumber(tokenCount * layoutStride * 4, 0)}`);
    }
    const data = {
      ...manifest,
      manifest,
      tokens,
      tokenCount,
      layoutStride,
      layout: new Float32Array(layoutBuffer),
      projection: manifest.projection ?? {},
      tensor: manifest.tensor ?? {
        name: "model.llm.embed.weight",
        shape: [tokenCount + Number(manifest.excludedPaddingRows ?? manifest.unmappedRows ?? 966), EMBEDDING_DIMENSIONS],
        dtype: "BF16",
        payloadBytes: Number(manifest.files?.vectors?.bytes ?? tokenCount * EMBEDDING_ROW_BYTES),
        unmappedRows: Number(manifest.excludedPaddingRows ?? manifest.unmappedRows ?? 966),
      },
      clusters: manifest.clusters ?? [],
    };
    state.embeddingData = data;
    dom.embeddingCount.textContent = `${formatNumber(tokenCount, 0)} / ${formatNumber(tokenCount, 0)} tokens mapped`;
    buildEmbeddingView(data);
    return data;
  })();
  try {
    return await embeddingLoadPromise;
  } catch (error) {
    embeddingLoadPromise = null;
    throw error;
  }
}

function buildEmbeddingView(data) {
  while (embeddingsGroup.children.length) disposeObject(embeddingsGroup.children.pop());
  embeddingFocusGroup = new THREE.Group();
  embeddingFocusGroup.name = "embedding-focus";
  const count = data.tokenCount;
  const stride = data.layoutStride;
  const source = data.layout;
  const declaredAxisScale = data.projection?.axisScaleP99 ?? data.projection?.axisScale ?? data.manifest?.axisScaleP99;
  const axisScale = Array.isArray(declaredAxisScale) && declaredAxisScale.length >= 3
    ? declaredAxisScale.slice(0, 3).map((value) => Math.max(Math.abs(Number(value)), 1e-8))
    : [1, 1, 1];
  const robustExtent = Number(data.projection?.visualExtent ?? 18);
  embeddingVisualPositions = new Float32Array(count * 3);
  embeddingNorms = new Float32Array(count);
  const specialFlags = new Uint8Array(count);
  const pointColors = new Uint8Array(count * 3);
  const pointColor = new THREE.Color();
  const pointBrightener = new THREE.Color(0xd9f2ff);
  const clusterPalette = (data.clusters?.length ? data.clusters : EMBEDDING_CLUSTER_COLORS).map((cluster, index) =>
    typeof cluster === "number" ? cluster : clusterColor(clusterIdentifier(cluster, index)),
  );
  for (let index = 0; index < count; index += 1) {
    const sourceOffset = index * stride;
    const targetOffset = index * 3;
    embeddingVisualPositions[targetOffset] = THREE.MathUtils.clamp(source[sourceOffset] / axisScale[0], -1.5, 1.5) * robustExtent;
    embeddingVisualPositions[targetOffset + 1] = THREE.MathUtils.clamp(source[sourceOffset + 1] / axisScale[1], -1.5, 1.5) * robustExtent;
    embeddingVisualPositions[targetOffset + 2] = THREE.MathUtils.clamp(source[sourceOffset + 2] / axisScale[2], -1.5, 1.5) * robustExtent;
    embeddingNorms[index] = source[sourceOffset + 3];
    specialFlags[index] = embeddingTokenType(data.tokens[index]).toLowerCase().includes("special") || embeddingTokenType(data.tokens[index]).toLowerCase().includes("control") ? 255 : 0;
    const clusterIndex = Math.abs(Number(data.tokens[index].cluster ?? 0)) % Math.max(clusterPalette.length, 1);
    pointColor.setHex(clusterPalette[clusterIndex] ?? EMBEDDING_CLUSTER_COLORS[0]).lerp(pointBrightener, 0.24);
    pointColors[targetOffset] = Math.round(pointColor.r * 255);
    pointColors[targetOffset + 1] = Math.round(pointColor.g * 255);
    pointColors[targetOffset + 2] = Math.round(pointColor.b * 255);
    data.tokens[index].norm = embeddingNorms[index];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(embeddingVisualPositions, 3));
  geometry.setAttribute("aSpecial", new THREE.BufferAttribute(specialFlags, 1, true));
  geometry.setAttribute("aColor", new THREE.BufferAttribute(pointColors, 3, true));
  geometry.computeBoundingSphere();
  embeddingOverviewSphere = geometry.boundingSphere?.clone() ?? new THREE.Sphere(new THREE.Vector3(0, 0, 0), robustExtent * 1.8);
  const material = new THREE.ShaderMaterial({
    uniforms: {
      uPixelRatio: { value: renderer.getPixelRatio() },
      uPointScale: { value: window.innerWidth < 720 ? 3.2 : 2.75 },
      uOpacity: { value: 0.74 },
    },
    vertexShader: `
      attribute float aSpecial;
      attribute vec3 aColor;
      uniform float uPixelRatio;
      uniform float uPointScale;
      varying float vSpecial;
      varying vec3 vColor;
      void main() {
        vSpecial = aSpecial;
        vColor = aColor;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        float perspective = 78.0 / max(18.0, -viewPosition.z);
        gl_PointSize = clamp(uPointScale * uPixelRatio * perspective, 1.4 * uPixelRatio, 7.2 * uPixelRatio);
      }
    `,
    fragmentShader: `
      uniform float uOpacity;
      varying float vSpecial;
      varying vec3 vColor;
      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        if (radius > 0.5) discard;
        float core = smoothstep(0.5, 0.07, radius);
        float halo = smoothstep(0.5, 0.20, radius);
        vec3 color = mix(vColor, vec3(0.94, 0.76, 0.34), vSpecial);
        gl_FragColor = vec4(color, (0.22 * halo + 0.78 * core) * uOpacity);
      }
    `,
    transparent: true,
    depthWrite: false,
    blending: THREE.NormalBlending,
  });
  embeddingPoints = new THREE.Points(geometry, material);
  embeddingPoints.userData.kind = "embedding-token";
  embeddingPoints.renderOrder = 2;
  embeddingsGroup.add(embeddingPoints);
  embeddingsGroup.add(embeddingFocusGroup);
  const anchorCandidates = new Map();
  const generalCandidates = [];
  for (const token of data.tokens) {
    const text = visibleToken(token).replaceAll("␠", " ").trim();
    const normalized = text.toLocaleLowerCase();
    const anchorRank = EMBEDDING_LABEL_ANCHOR_RANK.get(normalized);
    if (anchorRank !== undefined) {
      const variantPenalty = (token.leadingSpace ? 0 : 2) + (text === normalized ? 0 : 1);
      const current = anchorCandidates.get(normalized);
      if (!current || variantPenalty < current.variantPenalty || (variantPenalty === current.variantPenalty && Number(token.id) < Number(current.token.id))) {
        anchorCandidates.set(normalized, { token, anchorRank, variantPenalty });
      }
      continue;
    }
    const type = embeddingTokenType(token).toLowerCase();
    const cleanLatinWord = /^[A-Za-z][A-Za-z'’-]*$/.test(text);
    if (type === "word" && token.leadingSpace && cleanLatinWord && text.length >= 4 && text.length <= 16) {
      generalCandidates.push(token);
    }
  }
  const anchorIds = [...anchorCandidates.values()].sort((a, b) => a.anchorRank - b.anchorRank).map((entry) => Number(entry.token.id));
  embeddingAnchorCandidateCount = anchorIds.length;
  embeddingLabelCandidateIds = [
    ...anchorIds,
    ...generalCandidates.sort((a, b) => Number(a.id) - Number(b.id)).map((token) => Number(token.id)),
  ];
  scheduleEmbeddingLabelRefresh(true);
}

function setEmbeddingLabels(records) {
  dom.embeddingLabelLayer.replaceChildren();
  embeddingLabels = records.map((record) => {
    const element = document.createElement("div");
    element.className = `embedding-label ${record.kind === "cluster" ? "embedding-cluster-label" : "embedding-token-label"}`;
    element.dataset.rank = String(record.rank ?? 0);
    const title = document.createElement("strong");
    title.textContent = record.title;
    element.append(title);
    if (record.meta) {
      const meta = document.createElement("span");
      meta.textContent = record.meta;
      element.append(meta);
    }
    dom.embeddingLabelLayer.append(element);
    return { ...record, element };
  });
}

function projectedScreenPosition(position, matrix, width, height) {
  const e = matrix.elements;
  const x = position.x;
  const y = position.y;
  const z = position.z;
  const w = e[3] * x + e[7] * y + e[11] * z + e[15];
  if (w <= 0) return null;
  const ndcX = (e[0] * x + e[4] * y + e[8] * z + e[12]) / w;
  const ndcY = (e[1] * x + e[5] * y + e[9] * z + e[13]) / w;
  const ndcZ = (e[2] * x + e[6] * y + e[10] * z + e[14]) / w;
  if (ndcZ < -1 || ndcZ > 1 || ndcX < -1.04 || ndcX > 1.04 || ndcY < -1.04 || ndcY > 1.04) return null;
  return { x: (ndcX * 0.5 + 0.5) * width, y: (-ndcY * 0.5 + 0.5) * height };
}

function refreshEmbeddingOverviewLabels() {
  embeddingLabelRefreshTimer = null;
  if (state.mode !== "embeddings" || state.selectedToken || !state.embeddingData || state.embeddingLabelDensity === "off") {
    if (!state.selectedToken) setEmbeddingLabels([]);
    return;
  }
  const width = dom.viewport.clientWidth;
  const height = dom.viewport.clientHeight;
  const mobile = width < 720;
  const dense = state.embeddingLabelDensity === "dense";
  const limit = dense ? (mobile ? 56 : 180) : (mobile ? 24 : 90);
  const minimumCell = dense ? (mobile ? 48 : 42) : (mobile ? 74 : 62);
  const cellSize = Math.max(minimumCell, Math.sqrt((width * height) / Math.max(limit, 1)) * (dense ? 0.45 : 0.52));
  const matrix = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
  const occupied = new Set();
  const records = [];
  const position = new THREE.Vector3();
  const candidateLimit = dense ? embeddingLabelCandidateIds.length : embeddingAnchorCandidateCount;
  for (let candidateIndex = 0; candidateIndex < candidateLimit; candidateIndex += 1) {
    const tokenId = embeddingLabelCandidateIds[candidateIndex];
    const token = embeddingTokenFromReference(tokenId);
    const visual = embeddingVisualPosition(token, position);
    if (!visual) continue;
    const screen = projectedScreenPosition(visual, matrix, width, height);
    if (!screen) continue;
    const key = `${Math.floor(screen.x / cellSize)}:${Math.floor(screen.y / cellSize)}`;
    if (occupied.has(key)) continue;
    occupied.add(key);
    records.push({
      kind: "token",
      title: visibleToken(token),
      meta: `TOKEN ${tokenIdentifier(token)}`,
      position: visual.clone(),
      rank: records.length,
    });
    if (records.length >= limit) break;
  }
  setEmbeddingLabels(records);
}

function scheduleEmbeddingLabelRefresh(immediate = false) {
  if (embeddingLabelRefreshTimer !== null) window.clearTimeout(embeddingLabelRefreshTimer);
  if (immediate) refreshEmbeddingOverviewLabels();
  else embeddingLabelRefreshTimer = window.setTimeout(refreshEmbeddingOverviewLabels, 90);
}

function clearEmbeddingFocus() {
  while (embeddingFocusGroup?.children.length) disposeObject(embeddingFocusGroup.children.pop());
  if (embeddingPoints?.material?.uniforms?.uOpacity) embeddingPoints.material.uniforms.uOpacity.value = 0.74;
  scheduleEmbeddingLabelRefresh(true);
}

function float16ToFloat32(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

async function fetchBinaryRange(url, start, byteLength) {
  const end = start + byteLength - 1;
  const response = await fetch(url, { headers: { Range: `bytes=${start}-${end}` } });
  if (!response.ok) throw new Error(`${url} range request failed: ${response.status}`);
  const payload = await response.arrayBuffer();
  if (payload.byteLength === byteLength) return payload;
  if (payload.byteLength >= end + 1) return payload.slice(start, end + 1);
  throw new Error(`${url} returned ${formatNumber(payload.byteLength, 0)} bytes; expected ${formatNumber(byteLength, 0)}`);
}

async function fetchEmbeddingNeighbors(token) {
  const tokenId = Number(token?.id ?? token?.index);
  if (embeddingNeighborCache.has(tokenId)) return embeddingNeighborCache.get(tokenId);
  const request = (async () => {
    const count = manifestNeighborCount(state.embeddingData.manifest);
    const cosineDtype = String(
      embeddingManifestValue("neighborCosineDtype", "neighbourCosineDtype")
      ?? state.embeddingData.manifest.neighbors?.cosineDtype
      ?? state.embeddingData.manifest.neighbours?.cosineDtype
      ?? "float32",
    ).toLowerCase();
    const cosineBytes = cosineDtype.includes("16") ? 2 : 4;
    const [idsBuffer, cosineBuffer] = await Promise.all([
      fetchBinaryRange(EMBEDDING_NEIGHBOR_IDS_URL, tokenId * count * 4, count * 4),
      fetchBinaryRange(EMBEDDING_NEIGHBOR_COSINES_URL, tokenId * count * cosineBytes, count * cosineBytes),
    ]);
    const idsView = new DataView(idsBuffer);
    const cosineView = new DataView(cosineBuffer);
    const records = [];
    for (let rank = 0; rank < count; rank += 1) {
      const id = idsView.getUint32(rank * 4, true);
      if (id === 0xffffffff || id === tokenId || id >= state.embeddingData.tokenCount) continue;
      const cosine = cosineBytes === 2
        ? float16ToFloat32(cosineView.getUint16(rank * 2, true))
        : cosineView.getFloat32(rank * 4, true);
      if (!Number.isFinite(cosine)) continue;
      records.push({ id, index: id, cosine, rank: rank + 1, exact: true, global: true });
    }
    return records;
  })();
  embeddingNeighborCache.set(tokenId, request);
  try {
    return await request;
  } catch (error) {
    embeddingNeighborCache.delete(tokenId);
    throw error;
  }
}

function renderEmbeddingNeighborLens(token, neighborEntries = null, error = null) {
  if (!dom.embeddingNeighborLens) return;
  const tokenName = visibleToken(token);
  if (error) {
    dom.embeddingNeighborLens.innerHTML = `<div class="token-closeup"><header class="token-closeup-header"><div><span class="token-closeup-kicker">TOKEN CLOSE-UP</span><h3>${escapeHTML(tokenName)}</h3></div></header><div class="neighbor-lens-empty">${escapeHTML(error instanceof Error ? error.message : String(error))}</div></div>`;
    return;
  }
  if (!Array.isArray(neighborEntries)) {
    dom.embeddingNeighborLens.innerHTML = `<div class="token-closeup"><header class="token-closeup-header"><div><span class="token-closeup-kicker">TOKEN CLOSE-UP</span><h3>${escapeHTML(tokenName)}</h3><p>Token ${escapeHTML(tokenIdentifier(token))} · one vector with 6,144 learned values</p></div></header><div class="neighbor-lens-empty">Finding the 32 closest token pieces across all 200,058 entries…</div></div>`;
    return;
  }
  const entries = neighborEntries.slice(0, 32);
  const rows = entries.map(({ record, token: neighbor }, index) => {
    const score = Number(record.cosine);
    const width = `${Math.min(50, Math.abs(score) * 50).toFixed(3)}%`;
    const spacing = neighbor.leadingSpace ? "starts with a space" : "no starting space";
    const direction = score >= 0 ? "positive" : "negative";
    return `<li><button class="token-neighbor-item" type="button" data-token-index="${escapeHTML(String(neighbor.id))}" aria-label="Rank ${index + 1}. ${escapeHTML(visibleToken(neighbor))}. ${spacing}. Cosine similarity ${score.toFixed(6)}. Select this token."><span class="token-neighbor-rank">${String(index + 1).padStart(2, "0")}</span><span class="token-neighbor-name"><strong>${escapeHTML(visibleToken(neighbor))}</strong><small>${spacing}</small></span><span class="token-neighbor-cosine">${score.toFixed(6)}</span><span class="cosine-track" aria-hidden="true"><i class="${direction}" style="--cosine-width:${width}"></i></span></button></li>`;
  }).join("");
  dom.embeddingNeighborLens.innerHTML = `<div class="token-closeup">
    <header class="token-closeup-header">
      <div><span class="token-closeup-kicker">TOKEN CLOSE-UP</span><h3>${escapeHTML(tokenName)}</h3><p>Token ${escapeHTML(tokenIdentifier(token))} · one vector with 6,144 learned values</p></div>
      <button class="token-closeup-detail" type="button" data-action="open-vector-details">Vector details</button>
    </header>
    <div class="token-closeup-explanation">These are the 32 token pieces whose learned vectors point most nearly in the same direction as <strong>${escapeHTML(tokenName)}</strong>.</div>
    <div class="token-closeup-section"><strong>32 closest token pieces · exact across all 200,058 tokens</strong><span>Cosine: −1 opposite · 0 no directional alignment · +1 same direction</span></div>
    <ol class="token-neighbor-board" aria-label="32 closest token pieces">${rows}</ol>
    <footer class="token-closeup-note">Every printed number is calculated in all 6,144 dimensions. A close token is not automatically a synonym.</footer>
  </div>`;
}

function semanticColor(clusterId, alpha = 1) {
  const hue = (Number(clusterId) * 137.508 + 198) % 360;
  return `hsla(${hue.toFixed(1)}, 66%, 67%, ${alpha})`;
}

function ensureSemanticMapSurface() {
  if (semanticMapCanvas?.isConnected) return;
  dom.embeddingMeaningMap.innerHTML = `
    <div class="meaning-map-heading"><strong>Meaning map</strong><span>Similar token pieces gather together · each colour is one learned region</span></div>
    <canvas class="semantic-map-canvas" role="img" aria-label="Two-dimensional meaning map of all 200,058 Inkling tokenizer entries"></canvas>
    <button class="meaning-map-reset" type="button">Reset map</button>
    <div class="meaning-map-status" role="status">Preparing all 200,058 tokens…</div>`;
  semanticMapCanvas = dom.embeddingMeaningMap.querySelector(".semantic-map-canvas");
  semanticMapContext = semanticMapCanvas.getContext("2d", { alpha: true });
  semanticMapCanvas.addEventListener("pointerdown", (event) => {
    semanticMapCanvas.setPointerCapture(event.pointerId);
    semanticMapPointer = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      offsetX: semanticMapTransform.offsetX,
      offsetY: semanticMapTransform.offsetY,
      moved: false,
    };
  });
  semanticMapCanvas.addEventListener("pointermove", (event) => {
    if (!semanticMapPointer || semanticMapPointer.pointerId !== event.pointerId) return;
    const dx = event.clientX - semanticMapPointer.startX;
    const dy = event.clientY - semanticMapPointer.startY;
    if (Math.hypot(dx, dy) > 3) semanticMapPointer.moved = true;
    semanticMapTransform.offsetX = semanticMapPointer.offsetX + dx;
    semanticMapTransform.offsetY = semanticMapPointer.offsetY + dy;
    scheduleSemanticMapDraw();
  });
  semanticMapCanvas.addEventListener("pointerup", (event) => {
    if (!semanticMapPointer || semanticMapPointer.pointerId !== event.pointerId) return;
    const moved = semanticMapPointer.moved;
    semanticMapPointer = null;
    if (!moved) selectSemanticTokenAt(event.clientX, event.clientY);
  });
  semanticMapCanvas.addEventListener("pointercancel", () => { semanticMapPointer = null; });
  semanticMapCanvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    const rect = semanticMapCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const oldScale = semanticMapTransform.scale;
    const nextScale = THREE.MathUtils.clamp(oldScale * Math.exp(-event.deltaY * 0.00115), 0.72, 22);
    const centerX = rect.width * 0.5;
    const centerY = rect.height * 0.5;
    const localX = (x - centerX - semanticMapTransform.offsetX) / oldScale;
    const localY = (y - centerY - semanticMapTransform.offsetY) / oldScale;
    semanticMapTransform.scale = nextScale;
    semanticMapTransform.offsetX = x - centerX - localX * nextScale;
    semanticMapTransform.offsetY = y - centerY - localY * nextScale;
    scheduleSemanticMapDraw();
  }, { passive: false });
  dom.embeddingMeaningMap.querySelector(".meaning-map-reset").addEventListener("click", () => {
    semanticMapTransform.scale = 1;
    semanticMapTransform.offsetX = 0;
    semanticMapTransform.offsetY = 0;
    scheduleSemanticMapDraw();
  });
}

function robustSemanticBounds(layout, count) {
  const xs = new Float32Array(count);
  const ys = new Float32Array(count);
  for (let id = 0; id < count; id += 1) {
    xs[id] = layout[id * 2];
    ys[id] = layout[id * 2 + 1];
  }
  xs.sort();
  ys.sort();
  const low = Math.floor(count * 0.0025);
  const high = Math.min(count - 1, Math.ceil(count * 0.9975));
  return { minX: xs[low], maxX: xs[high], minY: ys[low], maxY: ys[high] };
}

function semanticTokenLabelCandidates(manifest) {
  const seen = new Set();
  const candidates = [];
  const add = (reference) => {
    const id = Number(reference);
    if (!Number.isInteger(id) || id < 0 || id >= EXPECTED_TOKEN_COUNT || seen.has(id)) return;
    const token = embeddingTokenFromReference(id);
    if (!token || !visibleToken(token).trim()) return;
    seen.add(id);
    candidates.push(id);
  };
  const clusters = [...(manifest.clusters ?? [])].sort((a, b) => Number(b.count) - Number(a.count));
  for (const cluster of clusters) add(cluster.representativeTokenIds?.[0]);
  const anchors = [];
  for (let tokenId = 0; tokenId < EXPECTED_TOKEN_COUNT; tokenId += 1) {
    const token = embeddingTokenFromReference(tokenId);
    const normalized = rawToken(token).replace(/^Ġ/, "").trim().toLowerCase();
    const rank = EMBEDDING_LABEL_ANCHOR_RANK.get(normalized);
    if (rank !== undefined) anchors.push({ tokenId, rank });
  }
  anchors.sort((a, b) => a.rank - b.rank);
  for (const anchor of anchors) add(anchor.tokenId);
  for (let representativeIndex = 1; representativeIndex < 5; representativeIndex += 1) {
    for (const cluster of clusters) add(cluster.representativeTokenIds?.[representativeIndex]);
  }
  return candidates;
}

async function loadSemanticMapData() {
  if (semanticMapData) return semanticMapData;
  if (semanticMapLoadPromise) return semanticMapLoadPromise;
  ensureSemanticMapSurface();
  semanticMapLoadPromise = (async () => {
    const [manifestResponse, layoutResponse, clustersResponse] = await Promise.all([
      fetch(SEMANTIC_MANIFEST_URL),
      fetch(SEMANTIC_LAYOUT_URL),
      fetch(SEMANTIC_CLUSTERS_URL),
    ]);
    if (!manifestResponse.ok) throw new Error(`Meaning-map manifest request failed: ${manifestResponse.status}`);
    if (!layoutResponse.ok) throw new Error(`Meaning-map layout request failed: ${layoutResponse.status}`);
    if (!clustersResponse.ok) throw new Error(`Meaning-map clusters request failed: ${clustersResponse.status}`);
    const [manifest, layoutBuffer, clustersBuffer] = await Promise.all([
      manifestResponse.json(),
      layoutResponse.arrayBuffer(),
      clustersResponse.arrayBuffer(),
    ]);
    const tokenCount = Number(manifest.tokenCount);
    if (tokenCount !== EXPECTED_TOKEN_COUNT) throw new Error(`Meaning map has ${formatNumber(tokenCount, 0)} tokens; expected ${formatNumber(EXPECTED_TOKEN_COUNT, 0)}`);
    if (layoutBuffer.byteLength !== tokenCount * 2 * 4) throw new Error("Meaning-map coordinate payload has the wrong size");
    if (clustersBuffer.byteLength !== tokenCount * 4) throw new Error("Meaning-map cluster payload has the wrong size");
    const layout = new Float32Array(layoutBuffer);
    const clusters = new Uint32Array(clustersBuffer);
    const clusterCount = Number(manifest.clusterCount);
    const counts = new Uint32Array(clusterCount);
    for (const clusterId of clusters) if (clusterId < clusterCount) counts[clusterId] += 1;
    const starts = new Uint32Array(clusterCount + 1);
    for (let clusterId = 0; clusterId < clusterCount; clusterId += 1) starts[clusterId + 1] = starts[clusterId] + counts[clusterId];
    const cursors = starts.slice(0, clusterCount);
    const tokenOrder = new Uint32Array(tokenCount);
    for (let tokenId = 0; tokenId < tokenCount; tokenId += 1) {
      const clusterId = clusters[tokenId];
      if (clusterId < clusterCount) tokenOrder[cursors[clusterId]++] = tokenId;
    }
    semanticMapData = {
      manifest,
      layout,
      clusters,
      clusterCount,
      starts,
      tokenOrder,
      bounds: robustSemanticBounds(layout, tokenCount),
      labelTokenIds: semanticTokenLabelCandidates(manifest),
    };
    const status = dom.embeddingMeaningMap.querySelector(".meaning-map-status");
    if (status) status.textContent = `${formatNumber(tokenCount, 0)} tokens · ${formatNumber(clusterCount, 0)} learned communities`;
    resizeSemanticMap();
    return semanticMapData;
  })();
  try {
    return await semanticMapLoadPromise;
  } catch (error) {
    semanticMapLoadPromise = null;
    throw error;
  }
}

function semanticMapGeometry() {
  if (!semanticMapCanvas || !semanticMapData) return null;
  const rect = semanticMapCanvas.getBoundingClientRect();
  const { minX, maxX, minY, maxY } = semanticMapData.bounds;
  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const fit = Math.min(Math.max(40, rect.width - 76) / spanX, Math.max(40, rect.height - 88) / spanY);
  return {
    rect,
    midX: (minX + maxX) * 0.5,
    midY: (minY + maxY) * 0.5,
    fit,
    centerX: rect.width * 0.5,
    centerY: rect.height * 0.5 + 8,
  };
}

function semanticCoordinateToScreen(x, y, geometry) {
  return {
    x: geometry.centerX + (x - geometry.midX) * geometry.fit * semanticMapTransform.scale + semanticMapTransform.offsetX,
    y: geometry.centerY - (y - geometry.midY) * geometry.fit * semanticMapTransform.scale + semanticMapTransform.offsetY,
  };
}

function semanticPointToScreen(tokenId, geometry) {
  return semanticCoordinateToScreen(
    semanticMapData.layout[tokenId * 2],
    semanticMapData.layout[tokenId * 2 + 1],
    geometry,
  );
}

function scheduleSemanticMapDraw() {
  if (semanticMapFrame !== null) return;
  semanticMapFrame = window.requestAnimationFrame(() => {
    semanticMapFrame = null;
    drawSemanticMap();
  });
}

function drawSemanticMap() {
  if (!semanticMapData || !semanticMapCanvas || state.embeddingView !== "meaning") return;
  const geometry = semanticMapGeometry();
  if (!geometry || geometry.rect.width <= 0 || geometry.rect.height <= 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.max(1, Math.round(geometry.rect.width * ratio));
  const pixelHeight = Math.max(1, Math.round(geometry.rect.height * ratio));
  if (semanticMapCanvas.width !== pixelWidth || semanticMapCanvas.height !== pixelHeight) {
    semanticMapCanvas.width = pixelWidth;
    semanticMapCanvas.height = pixelHeight;
  }
  const context = semanticMapContext;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, geometry.rect.width, geometry.rect.height);
  context.globalCompositeOperation = "source-over";
  const pointSize = THREE.MathUtils.clamp(1.18 + Math.log2(Math.max(1, semanticMapTransform.scale)) * 0.58, 1.1, 3.1);
  const margin = pointSize * 2;
  for (let clusterId = 0; clusterId < semanticMapData.clusterCount; clusterId += 1) {
    context.fillStyle = semanticColor(clusterId, semanticMapTransform.scale < 2 ? 0.52 : 0.68);
    const start = semanticMapData.starts[clusterId];
    const end = semanticMapData.starts[clusterId + 1];
    for (let position = start; position < end; position += 1) {
      const tokenId = semanticMapData.tokenOrder[position];
      const point = semanticPointToScreen(tokenId, geometry);
      if (point.x < -margin || point.y < -margin || point.x > geometry.rect.width + margin || point.y > geometry.rect.height + margin) continue;
      context.fillRect(point.x - pointSize * 0.5, point.y - pointSize * 0.5, pointSize, pointSize);
    }
  }
  context.globalCompositeOperation = "source-over";
  const clusterRecords = [...(semanticMapData.manifest.clusters ?? [])]
    .sort((a, b) => Number(b.count) - Number(a.count));
  const labelLimit = semanticMapTransform.scale < 1.55 ? 14 : semanticMapTransform.scale < 3 ? 18 : semanticMapTransform.scale < 5 ? 9 : 4;
  const occupiedLabels = [];
  let labelsDrawn = 0;
  context.font = '10px ui-monospace, "SFMono-Regular", Menlo, monospace';
  context.textBaseline = "middle";
  for (const cluster of clusterRecords) {
    if (labelsDrawn >= labelLimit) break;
    const centroid = Array.isArray(cluster.centroid) ? cluster.centroid : null;
    if (!centroid) continue;
    const point = semanticCoordinateToScreen(Number(centroid[0]), Number(centroid[1]), geometry);
    if (point.x < 48 || point.y < 34 || point.x > geometry.rect.width - 48 || point.y > geometry.rect.height - 34) continue;
    const rawLabel = String(cluster.label ?? cluster.representativeTokens?.slice(0, 3).join(" · ") ?? `Group ${cluster.id}`);
    const label = rawLabel.length > 30 ? `${rawLabel.slice(0, 29)}…` : rawLabel;
    const textWidth = context.measureText(label).width;
    const labelRect = { left: point.x - textWidth * 0.5 - 9, right: point.x + textWidth * 0.5 + 9, top: point.y - 13, bottom: point.y + 13 };
    if (occupiedLabels.some((other) => labelRect.left < other.right && labelRect.right > other.left && labelRect.top < other.bottom && labelRect.bottom > other.top)) continue;
    occupiedLabels.push(labelRect);
    context.fillStyle = "rgba(3, 8, 11, 0.84)";
    context.strokeStyle = semanticColor(cluster.id, 0.58);
    context.lineWidth = 1;
    context.beginPath();
    context.roundRect(point.x - textWidth * 0.5 - 7, point.y - 10, textWidth + 14, 20, 5);
    context.fill();
    context.stroke();
    context.fillStyle = "rgba(235, 244, 248, 0.94)";
    context.fillText(label, point.x - textWidth * 0.5, point.y + 0.5);
    labelsDrawn += 1;
  }
  if (semanticMapTransform.scale >= 2.15) {
    const tokenLabelLimit = semanticMapTransform.scale < 3.8 ? 24 : semanticMapTransform.scale < 7 ? 48 : 82;
    let tokenLabelsDrawn = 0;
    context.font = '9px ui-monospace, "SFMono-Regular", Menlo, monospace';
    for (const tokenId of semanticMapData.labelTokenIds ?? []) {
      if (tokenLabelsDrawn >= tokenLabelLimit) break;
      const point = semanticPointToScreen(tokenId, geometry);
      if (point.x < 18 || point.y < 24 || point.x > geometry.rect.width - 88 || point.y > geometry.rect.height - 24) continue;
      const token = embeddingTokenFromReference(tokenId);
      const rawLabel = visibleToken(token);
      const label = rawLabel.length > 22 ? `${rawLabel.slice(0, 21)}…` : rawLabel;
      const textWidth = context.measureText(label).width;
      const labelRect = { left: point.x + 6, right: point.x + textWidth + 18, top: point.y - 10, bottom: point.y + 10 };
      if (occupiedLabels.some((other) => labelRect.left < other.right && labelRect.right > other.left && labelRect.top < other.bottom && labelRect.bottom > other.top)) continue;
      occupiedLabels.push(labelRect);
      context.fillStyle = "rgba(5, 13, 17, 0.88)";
      context.fillRect(labelRect.left, labelRect.top, labelRect.right - labelRect.left, labelRect.bottom - labelRect.top);
      context.fillStyle = "rgba(127, 222, 249, 0.96)";
      context.beginPath();
      context.arc(point.x, point.y, 2.4, 0, Math.PI * 2);
      context.fill();
      context.fillStyle = "rgba(236, 246, 250, 0.95)";
      context.fillText(label, point.x + 11, point.y + 0.5);
      tokenLabelsDrawn += 1;
    }
  }
  if (state.selectedToken) {
    const selectedId = Number(state.selectedToken.id ?? state.selectedToken.index);
    const point = semanticPointToScreen(selectedId, geometry);
    context.strokeStyle = "rgba(255,255,255,0.96)";
    context.lineWidth = 2;
    context.beginPath();
    context.arc(point.x, point.y, 7, 0, Math.PI * 2);
    context.stroke();
  }
}

function resizeSemanticMap() {
  if (!semanticMapCanvas || !semanticMapData) return;
  scheduleSemanticMapDraw();
}

function selectSemanticTokenAt(clientX, clientY) {
  if (!semanticMapData || !semanticMapCanvas) return;
  const geometry = semanticMapGeometry();
  const x = clientX - geometry.rect.left;
  const y = clientY - geometry.rect.top;
  let bestId = -1;
  let bestDistance = semanticMapTransform.scale >= 4 ? 18 * 18 : 12 * 12;
  for (let tokenId = 0; tokenId < EXPECTED_TOKEN_COUNT; tokenId += 1) {
    const point = semanticPointToScreen(tokenId, geometry);
    const distance = (point.x - x) ** 2 + (point.y - y) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestId = tokenId;
    }
  }
  if (bestId >= 0) selectEmbeddingToken(embeddingTokenFromReference(bestId), { openNearby: true });
}

function renderSemanticMapError(error) {
  ensureSemanticMapSurface();
  const status = dom.embeddingMeaningMap.querySelector(".meaning-map-status");
  if (status) status.textContent = error instanceof Error ? error.message : String(error);
}

function syncEmbeddingView() {
  const inEmbeddings = state.mode === "embeddings";
  state.embeddingView = "all";
  dom.embeddingViewButtons.forEach((button) => {
    button.classList.add("is-active");
    button.setAttribute("aria-pressed", "true");
  });
  dom.embeddingControl.classList.remove("is-closeup");
  dom.embeddingNeighborLens.hidden = true;
  dom.embeddingMeaningMap.hidden = true;
  dom.embeddingLabelLayer.hidden = !inEmbeddings;
  if (embeddingFocusGroup) embeddingFocusGroup.visible = inEmbeddings;
  if (embeddingPoints?.material?.uniforms?.uOpacity) {
    embeddingPoints.material.uniforms.uOpacity.value = state.selectedToken ? 0.055 : 0.74;
  }
  if (inEmbeddings) {
    const geometryGroups = state.embeddingData?.clusters?.length ?? state.embeddingData?.projection?.geometryGroups ?? "";
    dom.sceneKey.innerHTML = `<span><i style="--key:#8295ad"></i>All 200,058 tokens${geometryGroups ? ` · ${formatNumber(geometryGroups, 0)} geometry groups` : ""}</span><span><i style="--key:#ffffff"></i>Selected vector</span><span><i style="--key:#73d7f5"></i>Exact global neighbours</span><span><i style="--key:#f0c56a"></i>Special / control</span>`;
    dom.axisGuide.innerHTML = '<span>X / Y / Z · PCA coordinates</span><span>Edges · original 6,144D cosine</span>';
    dom.viewInstructions.innerHTML = '<span>Search any tokenizer entry or click a point</span><span>Drag to orbit</span><span>Spokes show exhaustive global neighbours</span>';
  }
}

function setEmbeddingView(view, { pushHistory = true } = {}) {
  if (view !== "all") return;
  state.embeddingView = "all";
  syncEmbeddingView();
  if (state.selectedToken) focusEmbeddingToken(state.selectedToken, embeddingActiveNeighbors);
  else focusEmbeddings();
  renderCurrentInspector();
  updateURL(pushHistory);
}

function focusPointMaterial() {
  return new THREE.ShaderMaterial({
    uniforms: { uPixelRatio: { value: renderer.getPixelRatio() } },
    vertexShader: `
      attribute float aSelected;
      uniform float uPixelRatio;
      varying float vSelected;
      void main() {
        vSelected = aSelected;
        vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
        gl_Position = projectionMatrix * viewPosition;
        float perspective = 88.0 / max(16.0, -viewPosition.z);
        float base = mix(8.5, 15.0, aSelected);
        gl_PointSize = clamp(base * uPixelRatio * perspective, mix(7.0, 13.0, aSelected) * uPixelRatio, mix(17.0, 27.0, aSelected) * uPixelRatio);
      }
    `,
    fragmentShader: `
      varying float vSelected;
      void main() {
        float radius = length(gl_PointCoord - vec2(0.5));
        if (radius > 0.5) discard;
        float halo = smoothstep(0.5, 0.0, radius);
        float core = smoothstep(0.24, 0.0, radius);
        vec3 color = mix(vec3(0.40, 0.86, 1.0), vec3(1.0), vSelected);
        gl_FragColor = vec4(color, min(1.0, 0.44 * halo + 0.92 * core));
      }
    `,
    transparent: true,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
  });
}

function cylinderBetween(start, end, radius, material, radialSegments = 8) {
  const direction = end.clone().sub(start);
  const length = direction.length();
  if (length <= 1e-6) return null;
  const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius, radius, 1, radialSegments, 1, true), material);
  mesh.position.copy(start).add(end).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  mesh.scale.set(1, length, 1);
  mesh.frustumCulled = false;
  return mesh;
}

function instancedCylinders(segments, radius, material, renderOrder) {
  if (!segments.length) return null;
  const geometry = new THREE.CylinderGeometry(radius, radius, 1, 6, 1, true);
  const mesh = new THREE.InstancedMesh(geometry, material, segments.length);
  const transform = new THREE.Object3D();
  const up = new THREE.Vector3(0, 1, 0);
  segments.forEach((segment, index) => {
    const direction = segment.end.clone().sub(segment.start);
    const length = direction.length();
    transform.position.copy(segment.start).add(segment.end).multiplyScalar(0.5);
    transform.quaternion.setFromUnitVectors(up, direction.normalize());
    transform.scale.set(1, length, 1);
    transform.updateMatrix();
    mesh.setMatrixAt(index, transform.matrix);
    mesh.setColorAt(index, segment.color);
  });
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.frustumCulled = false;
  mesh.renderOrder = renderOrder;
  return mesh;
}

function addLuminousProjectedVector(group, end) {
  const origin = new THREE.Vector3(0, 0, 0);
  const direction = end.clone();
  const length = direction.length();
  if (length <= 0.15) return;
  direction.normalize();
  const headLength = THREE.MathUtils.clamp(length * 0.10, 1.05, 2.7);
  const shaftEnd = end.clone().addScaledVector(direction, -headLength * 0.82);
  const outerMaterial = new THREE.MeshBasicMaterial({
    color: 0x6fddff,
    transparent: true,
    opacity: 0.17,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  });
  const coreMaterial = new THREE.MeshBasicMaterial({
    color: 0xf4fdff,
    transparent: true,
    opacity: 0.93,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  });
  const glowShaft = cylinderBetween(origin, shaftEnd, 0.22, outerMaterial, 12);
  const coreShaft = cylinderBetween(origin, shaftEnd, 0.065, coreMaterial, 10);
  if (glowShaft) {
    glowShaft.renderOrder = 27;
    group.add(glowShaft);
  }
  if (coreShaft) {
    coreShaft.renderOrder = 28;
    group.add(coreShaft);
  }
  const addHead = (radius, headHeight, material, order) => {
    const cone = new THREE.Mesh(new THREE.ConeGeometry(radius, headHeight, 14, 1, true), material);
    cone.position.copy(end).addScaledVector(direction, -headHeight * 0.5);
    cone.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    cone.frustumCulled = false;
    cone.renderOrder = order;
    group.add(cone);
  };
  addHead(0.64, headLength * 1.12, outerMaterial.clone(), 27);
  addHead(0.34, headLength, coreMaterial.clone(), 28);
  const originGlow = new THREE.Mesh(
    new THREE.SphereGeometry(0.24, 12, 8),
    new THREE.MeshBasicMaterial({ color: 0xa8efff, transparent: true, opacity: 0.8, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending, toneMapped: false }),
  );
  originGlow.renderOrder = 28;
  group.add(originGlow);
}

async function buildEmbeddingFocus(token, generation = embeddingSelectionGeneration) {
  if (!embeddingPoints || !embeddingFocusGroup) return [];
  const neighborRecords = (await fetchEmbeddingNeighbors(token)).slice(0, 32)
    .map((record) => ({ record, token: neighborToken(record) }))
    .filter((entry) => entry.token && embeddingVisualPosition(entry.token));
  if (generation !== embeddingSelectionGeneration || state.selectedToken !== token) return [];
  while (embeddingFocusGroup.children.length) disposeObject(embeddingFocusGroup.children.pop());
  embeddingPoints.material.uniforms.uOpacity.value = 0.055;
  const selectedPosition = embeddingVisualPosition(token);
  if (!selectedPosition) return [];
  const focusGeometry = new THREE.BufferGeometry();
  const focusPositions = [selectedPosition, ...neighborRecords.map((entry) => embeddingVisualPosition(entry.token))];
  focusGeometry.setFromPoints(focusPositions);
  const selectedFlags = new Float32Array(focusPositions.length);
  selectedFlags[0] = 1;
  focusGeometry.setAttribute("aSelected", new THREE.BufferAttribute(selectedFlags, 1));
  const focusMaterial = focusPointMaterial();
  const focusPoints = new THREE.Points(focusGeometry, focusMaterial);
  focusPoints.renderOrder = 30;
  embeddingFocusGroup.add(focusPoints);
  const edgeSegments = neighborRecords.map(({ record, token: neighbor }) => {
    const strength = THREE.MathUtils.clamp((Number(record.cosine) - 0.04) / 0.56, 0.22, 1);
    return {
      start: selectedPosition,
      end: embeddingVisualPosition(neighbor),
      color: new THREE.Color(0x46b9dc).lerp(new THREE.Color(0xd5f8ff), strength * 0.68),
    };
  });
  const edgeGlow = instancedCylinders(edgeSegments, 0.13, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
    transparent: true,
    opacity: 0.12,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  }), 24);
  const edgeCore = instancedCylinders(edgeSegments, 0.045, new THREE.MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
    transparent: true,
    opacity: 0.74,
    depthWrite: false,
    depthTest: false,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
  }), 25);
  if (edgeGlow) embeddingFocusGroup.add(edgeGlow);
  if (edgeCore) embeddingFocusGroup.add(edgeCore);
  addLuminousProjectedVector(embeddingFocusGroup, selectedPosition);
  setEmbeddingLabels([
    { kind: "token", title: visibleToken(token), meta: `TOKEN ${tokenIdentifier(token)} · PROJECTED VECTOR`, position: selectedPosition, rank: 0 },
    ...neighborRecords.map(({ record, token: neighbor }, index) => ({
      kind: "token",
      title: visibleToken(neighbor),
      meta: `${Number(record.cosine).toFixed(4)} cosine · GLOBAL`,
      position: embeddingVisualPosition(neighbor),
      rank: index + 1,
    })),
  ]);
  return neighborRecords;
}

function updateEmbeddingLabels() {
  if (state.mode !== "embeddings" || !embeddingLabels.length || !camera) return;
  const width = dom.viewport.clientWidth;
  const height = dom.viewport.clientHeight;
  const focusMode = Boolean(state.selectedToken);
  const focusLabelCap = width < 720 ? 8 : 15;
  const occupied = [];
  let visibleFocusLabels = 0;
  embeddingLabels.forEach(({ element, position, rank = 0 }) => {
    const projected = position.clone().project(camera);
    let visible = projected.z > -1 && projected.z < 1 && projected.x > -1.08 && projected.x < 1.08 && projected.y > -1.08 && projected.y < 1.08;
    if (!visible) {
      element.hidden = true;
      return;
    }
    const x = (projected.x * 0.5 + 0.5) * width;
    const y = (-projected.y * 0.5 + 0.5) * height;
    element.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -120%)`;
    element.hidden = false;
    if (focusMode) {
      const isSelected = Number(rank) === 0;
      const labelWidth = Math.max(element.offsetWidth, isSelected ? 110 : 82);
      const labelHeight = Math.max(element.offsetHeight, 31);
      const offsets = isSelected ? [[0, 0]] : [
        [0, 0], [0, -34], [0, 34], [62, 0], [-62, 0], [52, -30], [-52, -30], [52, 30], [-52, 30],
        [102, 0], [-102, 0], [88, -42], [-88, -42], [88, 42], [-88, 42],
      ];
      let placement = null;
      if (isSelected || visibleFocusLabels < focusLabelCap) {
        for (const [offsetX, offsetY] of offsets) {
          const labelX = x + offsetX;
          const labelY = y + offsetY;
          const bounds = {
            left: labelX - labelWidth * 0.5 - 3,
            right: labelX + labelWidth * 0.5 + 3,
            top: labelY - labelHeight * 1.24 - 3,
            bottom: labelY - labelHeight * 0.10 + 3,
          };
          const onScreen = bounds.left >= 4 && bounds.right <= width - 4 && bounds.top >= 4 && bounds.bottom <= height - 4;
          const collides = occupied.some((other) => bounds.left < other.right && bounds.right > other.left && bounds.top < other.bottom && bounds.bottom > other.top);
          if (onScreen && !collides) {
            placement = { labelX, labelY, bounds };
            break;
          }
        }
      }
      if (!placement) visible = false;
      if (placement) {
        element.style.transform = `translate3d(${placement.labelX}px, ${placement.labelY}px, 0) translate(-50%, -120%)`;
        occupied.push(placement.bounds);
        if (!isSelected) visibleFocusLabels += 1;
      }
      element.hidden = !visible;
    }
  });
}

function disposeObject(object) {
  if (!object) return;
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    const disposeMaterial = (material) => {
      material?.map?.dispose?.();
      material?.dispose?.();
    };
    if (Array.isArray(child.material)) child.material.forEach(disposeMaterial);
    else disposeMaterial(child.material);
  });
}

function applyModeVisibility() {
  if (!architectureGroup) return;
  architectureGroup.visible = state.mode === "overview";
  tensorGroup.visible = state.mode === "overview";
  expertDetailGroup.visible = state.mode === "experts";
  embeddingsGroup.visible = state.mode === "embeddings";
  valuesGroup.visible = state.mode === "values";
  selectionBox.visible = state.mode === "overview" && Boolean(state.selectedTensor);
  hoverBox.visible = false;
  dom.modeButtons.forEach((button) => {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  dom.sceneKey.hidden = !["overview", "embeddings"].includes(state.mode);
  dom.layerControl.hidden = ["values", "embeddings"].includes(state.mode);
  dom.embeddingControl.hidden = state.mode !== "embeddings";
  dom.embeddingLabelLayer.hidden = state.mode !== "embeddings";
  if (state.mode === "overview") {
    dom.searchLabel.textContent = "Find a layer or tensor";
    dom.search.placeholder = "Try layer 29, gate.bias, audio…";
    dom.sceneKey.innerHTML = '<span><i style="--key:#7aa2ff"></i>Attention</span><span><i style="--key:#aa78ff"></i>Routed experts</span><span><i style="--key:#f0c56a"></i>Shared experts</span><span><i style="--key:#66d9a8"></i>Vision</span><span><i style="--key:#ff816c"></i>Audio</span>';
    dom.axisGuide.innerHTML = '<span>X · decoder depth</span><span>Y · operation path</span><span>Z · head / expert index</span>';
    dom.viewInstructions.innerHTML = "<span>Drag to orbit</span><span>Wheel or pinch to zoom</span><span>Click a tensor to inspect</span>";
  } else if (state.mode === "experts") {
    dom.searchLabel.textContent = "Find a layer or tensor";
    dom.search.placeholder = "Try layer 29, gate.bias, audio…";
    dom.axisGuide.innerHTML = '<span>X · learned gate bias</span><span>Y/Z · expert ID grid</span>';
    dom.viewInstructions.innerHTML = "<span>Each cell is one routed expert</span><span>Cyan marks the six highest stored biases</span>";
  } else if (state.mode === "embeddings") {
    const geometryGroups = state.embeddingData?.clusters?.length ?? state.embeddingData?.projection?.geometryGroups ?? "";
    dom.searchLabel.textContent = "Find a token or token ID";
    dom.search.placeholder = "Search all 200,058 tokens or enter an ID…";
    dom.sceneKey.innerHTML = `<span><i style="--key:#8295ad"></i>All 200,058 tokens${geometryGroups ? ` · ${formatNumber(geometryGroups, 0)} geometry groups` : ""}</span><span><i style="--key:#ffffff"></i>Selected vector</span><span><i style="--key:#73d7f5"></i>Exact global neighbours</span><span><i style="--key:#f0c56a"></i>Special / control</span>`;
    dom.axisGuide.innerHTML = '<span>X / Y / Z · PCA coordinates</span><span>Edges · original 6,144D cosine</span>';
    dom.viewInstructions.innerHTML = "<span>Search any tokenizer entry or click a point</span><span>Drag to orbit</span><span>Spokes show exhaustive global neighbours</span>";
  } else {
    dom.searchLabel.textContent = "Find a layer or tensor";
    dom.search.placeholder = "Try layer 29, gate.bias, audio…";
    dom.axisGuide.innerHTML = '<span>X · column index</span><span>Y · stored weight value</span><span>Z · row index</span>';
    dom.viewInstructions.innerHTML = "<span>Blue is negative</span><span>Coral is positive</span><span>Click a bar for its exact value</span>";
  }
  syncEmbeddingView();
}

async function setMode(mode, { pushHistory = true } = {}) {
  if (!state.data || !["overview", "experts", "embeddings", "values"].includes(mode)) return;
  state.mode = mode;
  state.hovered = null;
  dom.tooltip.hidden = true;
  dom.search.value = "";
  search("");
  applyModeVisibility();
  if (mode === "experts") {
    if (state.selectedLayer < 2) state.selectedLayer = 2;
    dom.layerSlider.value = String(state.selectedLayer);
    buildExpertDetail(state.data, state.selectedLayer);
    focusExperts(state.selectedLayer);
    renderLayerInspector(state.selectedLayer);
  } else if (mode === "values") {
    if (!state.activeValueSample && state.pendingTokenId !== null) {
      try {
        await loadEmbeddingData();
        const pending = embeddingTokenFromReference(state.pendingTokenId);
        state.pendingTokenId = null;
        if (pending) await prepareEmbeddingVectorValues(pending);
      } catch (error) {
        state.pendingTokenId = null;
        console.error(error);
      }
    }
    const sample = sampleForCurrentSelection();
    buildValueView(state.data, sample);
    focusValues();
    renderValueInspector(sample);
  } else if (mode === "embeddings") {
    dom.inspectorKicker.textContent = "TOKEN EMBEDDINGS";
    dom.inspectorTitle.textContent = "Loading complete learned geometry";
    dom.inspectorBody.innerHTML = '<div class="method-note">Reading all 200,058 projected token vectors and the complete vocabulary index.</div>';
    try {
      await loadEmbeddingData();
      if (state.mode !== "embeddings") return;
      const pending = state.pendingTokenId !== null ? embeddingTokenFromReference(state.pendingTokenId) : state.selectedToken;
      state.pendingTokenId = null;
      if (pending) selectEmbeddingToken(pending, { pushHistory: false, focus: true });
      else {
        clearEmbeddingFocus();
        focusEmbeddings();
        renderEmbeddingOverviewInspector();
      }
    } catch (error) {
      dom.inspectorKicker.textContent = "TOKEN EMBEDDINGS";
      dom.inspectorTitle.textContent = "Embedding map unavailable";
      dom.inspectorBody.innerHTML = `<div class="method-note">${escapeHTML(error instanceof Error ? error.message : String(error))}</div>`;
    }
  } else {
    if (state.selectedTensor) focusTensor(state.selectedTensor);
    else resetCamera();
    renderCurrentInspector();
  }
  applyModeVisibility();
  syncEmbeddingView();
  updateURL(pushHistory);
}

function cameraTo(position, target, duration = 650) {
  if (reducedMotion || duration <= 0) {
    camera.position.copy(position);
    controls.target.copy(target);
    controls.update();
    cameraTransition = null;
    return;
  }
  cameraTransition = {
    start: performance.now(),
    duration,
    fromPosition: camera.position.clone(),
    toPosition: position.clone(),
    fromTarget: controls.target.clone(),
    toTarget: target.clone(),
  };
}

function updateCameraTransition(now) {
  if (!cameraTransition) return;
  const elapsed = (now - cameraTransition.start) / cameraTransition.duration;
  const t = elapsed >= 1 ? 1 : 1 - Math.pow(1 - elapsed, 3);
  camera.position.lerpVectors(cameraTransition.fromPosition, cameraTransition.toPosition, t);
  controls.target.lerpVectors(cameraTransition.fromTarget, cameraTransition.toTarget, t);
  if (t >= 1) cameraTransition = null;
}

function resetCamera() {
  const mobile = dom.viewport.clientWidth < 720;
  const position = mobile ? new THREE.Vector3(98, 54, 112) : new THREE.Vector3(72, 39, 78);
  cameraTo(position, new THREE.Vector3(0, -3.15, 0), 720);
}

function focusLayer(layerId) {
  const x = layerX(layerId);
  cameraTo(new THREE.Vector3(x + 13.5, 8.4, 17.5), new THREE.Vector3(x, -0.2, 0), 620);
}

function focusTensor(tensor) {
  const index = state.data.tensors.indexOf(tensor);
  const layout = tensorLayouts[index];
  if (!layout) return;
  const offset = tensor.category === "attention" ? new THREE.Vector3(9, 6, 11) : new THREE.Vector3(9, 4, 12);
  cameraTo(layout.position.clone().add(offset), layout.position, 620);
}

function focusExperts(layerId) {
  const x = layerX(layerId);
  cameraTo(new THREE.Vector3(x + 13.5, -0.8, 13.5), new THREE.Vector3(x, -2.8, 0), 620);
}

function focusValues() {
  const mobile = dom.viewport.clientWidth < 720;
  cameraTo(mobile ? new THREE.Vector3(22, 18, 25) : new THREE.Vector3(12.5, 11.5, 14.5), new THREE.Vector3(0, 0, 0), 620);
}

function focusEmbeddings() {
  const radius = embeddingOverviewSphere?.radius ?? 29;
  const center = embeddingOverviewSphere?.center ?? new THREE.Vector3(0, 0, 0);
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * Math.max(camera.aspect, 0.1));
  const limitingHalfFov = Math.max(THREE.MathUtils.degToRad(8), Math.min(verticalHalfFov, horizontalHalfFov));
  const distance = (radius / Math.sin(limitingHalfFov)) * 1.08;
  const direction = new THREE.Vector3(1, 0.62, 1.12).normalize();
  controls.maxDistance = Math.max(220, distance * 1.65);
  cameraTo(center.clone().add(direction.multiplyScalar(distance)), center, 720);
}

function focusEmbeddingToken(token, neighborEntries = []) {
  const center = embeddingVisualPosition(token);
  if (!center) return;
  const neighbors = neighborEntries.map((entry) => entry.token ?? neighborToken(entry.record ?? entry)).filter(Boolean);
  const points = [new THREE.Vector3(0, 0, 0), center, ...neighbors.map((neighbor) => embeddingVisualPosition(neighbor)).filter(Boolean)];
  const sphere = new THREE.Sphere();
  new THREE.Box3().setFromPoints(points).getBoundingSphere(sphere);
  const radius = Math.max(sphere.radius, 2.4);
  const currentDirection = camera.position.clone().sub(controls.target).normalize();
  const verticalHalfFov = THREE.MathUtils.degToRad(camera.fov * 0.5);
  const horizontalHalfFov = Math.atan(Math.tan(verticalHalfFov) * Math.max(camera.aspect, 0.1));
  const fitHalfFov = Math.max(THREE.MathUtils.degToRad(9), Math.min(verticalHalfFov, horizontalHalfFov));
  const distance = Math.max(radius / Math.sin(fitHalfFov), radius * 2.8) * 1.12;
  cameraTo(sphere.center.clone().add(currentDirection.multiplyScalar(distance)), sphere.center, 620);
}

function setSelectionBox(layout, target = selectionBox) {
  if (!layout) {
    target.visible = false;
    return;
  }
  target.position.copy(layout.position);
  target.scale.copy(layout.scale).multiplyScalar(1.55);
  target.visible = true;
}

function findTensorIndex(tensor) {
  return state.data.tensors.findIndex((candidate) => candidate.name === tensor.name);
}

function selectTensor(tensor, { pushHistory = true, focus = true } = {}) {
  state.selectedTensor = tensor;
  state.selectedToken = null;
  state.selectedExpert = null;
  state.selectedValue = null;
  state.activeValueSample = null;
  if (tensor.layer !== null) {
    state.selectedLayer = tensor.layer;
    dom.layerSlider.value = String(tensor.layer);
  }
  const index = findTensorIndex(tensor);
  setSelectionBox(tensorLayouts[index]);
  setMode("overview", { pushHistory: false });
  if (focus) focusTensor(tensor);
  renderTensorInspector(tensor);
  updateLayerLabels();
  openInspector();
  updateURL(pushHistory);
}

function selectLayer(layerId, { pushHistory = true, focus = true } = {}) {
  state.selectedLayer = THREE.MathUtils.clamp(layerId, 0, state.data.layers.length - 1);
  state.selectedTensor = null;
  state.selectedToken = null;
  state.selectedExpert = null;
  selectionBox.visible = false;
  dom.layerSlider.value = String(state.selectedLayer);
  updateLayerLabels();
  buildExpertDetail(state.data, state.selectedLayer);
  renderLayerInspector(state.selectedLayer);
  if (focus) {
    if (state.mode === "experts") focusExperts(state.selectedLayer);
    else focusLayer(state.selectedLayer);
  }
  openInspector();
  updateURL(pushHistory);
}

function selectExpert(expertId) {
  const detail = expertInstanceData[expertId];
  if (!detail) return;
  state.selectedExpert = detail;
  renderExpertInspector(detail);
  openInspector();
  updateURL(true);
}

function selectValue(instanceId) {
  const detail = valueInstanceData[instanceId];
  if (!detail) return;
  state.selectedValue = detail;
  renderValueInspector(sampleForCurrentSelection(), detail);
  openInspector();
}

function selectEmbeddingToken(reference, { pushHistory = true, focus = true } = {}) {
  const token = embeddingTokenFromReference(reference);
  if (!token) return;
  const generation = ++embeddingSelectionGeneration;
  state.selectedToken = token;
  state.selectedValue = null;
  embeddingActiveNeighbors = [];
  state.embeddingView = "all";
  while (embeddingFocusGroup?.children.length) disposeObject(embeddingFocusGroup.children.pop());
  if (embeddingPoints?.material?.uniforms?.uOpacity) embeddingPoints.material.uniforms.uOpacity.value = 0.18;
  setEmbeddingLabels([{ kind: "token", title: visibleToken(token), meta: `TOKEN ${tokenIdentifier(token)} · PROJECTED VECTOR`, position: embeddingVisualPosition(token), rank: 0 }]);
  syncEmbeddingView();
  renderEmbeddingTokenInspector(token, null);
  openInspector();
  updateURL(pushHistory);
  buildEmbeddingFocus(token, generation).then((neighbors) => {
    if (generation !== embeddingSelectionGeneration || state.selectedToken !== token) return;
    embeddingActiveNeighbors = neighbors;
    syncEmbeddingView();
    if (focus) focusEmbeddingToken(token, neighbors);
    renderEmbeddingTokenInspector(token, neighbors);
  }).catch((error) => {
    if (generation !== embeddingSelectionGeneration || state.selectedToken !== token) return;
    embeddingActiveNeighbors = [];
    syncEmbeddingView();
    renderEmbeddingTokenInspector(token, [], error);
  });
}

function clearEmbeddingSelection({ pushHistory = true, focus = false } = {}) {
  embeddingSelectionGeneration += 1;
  state.selectedToken = null;
  embeddingActiveNeighbors = [];
  state.embeddingView = "all";
  dom.embeddingNeighborLens.replaceChildren();
  clearEmbeddingFocus();
  syncEmbeddingView();
  renderEmbeddingOverviewInspector();
  if (focus) focusEmbeddings();
  updateURL(pushHistory);
}

async function fetchEmbeddingVector(token) {
  const cacheKey = tokenIdentifier(token);
  if (embeddingVectorCache.has(cacheKey)) return embeddingVectorCache.get(cacheKey);
  const request = (async () => {
    const tokenId = Number(token.id ?? token.index);
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= state.embeddingData.tokenCount) throw new Error("Token ID is outside the full vocabulary");
    let rowBuffer;
    try {
      rowBuffer = await fetchBinaryRange(EMBEDDING_VECTOR_URL, tokenId * EMBEDDING_ROW_BYTES, EMBEDDING_ROW_BYTES);
    } catch (error) {
      throw new Error("The optional 2.46 GB full-vector payload is not installed. Run `npm run download:embeddings:full`, then reload the atlas.", { cause: error });
    }
    const source = new DataView(rowBuffer);
    const scratch = new DataView(new ArrayBuffer(4));
    const values = new Float32Array(EMBEDDING_DIMENSIONS);
    for (let index = 0; index < EMBEDDING_DIMENSIONS; index += 1) {
      const bf16 = source.getUint16(index * 2, true);
      scratch.setUint32(0, bf16 << 16, true);
      values[index] = scratch.getFloat32(0, true);
    }
    return values;
  })();
  embeddingVectorCache.set(cacheKey, request);
  try {
    return await request;
  } catch (error) {
    embeddingVectorCache.delete(cacheKey);
    throw error;
  }
}

function vectorSignatureHTML(values) {
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let squared = 0;
  let positive = 0;
  let negative = 0;
  const strongest = [];
  values.forEach((value, dimension) => {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
    squared += value * value;
    if (value > 0) positive += 1;
    else if (value < 0) negative += 1;
    const magnitude = Math.abs(value);
    if (strongest.length < 6 || magnitude > strongest.at(-1).magnitude) {
      strongest.push({ dimension, value, magnitude });
      strongest.sort((a, b) => b.magnitude - a.magnitude);
      strongest.length = Math.min(6, strongest.length);
    }
  });
  const norm = Math.sqrt(squared);
  return `
    <dl class="detail-list">
      <div class="detail-row"><dt>Decoded norm</dt><dd>${norm.toPrecision(8)}</dd></div>
      <div class="detail-row"><dt>Minimum / maximum</dt><dd>${minimum.toPrecision(6)} / ${maximum.toPrecision(6)}</dd></div>
      <div class="detail-row"><dt>Mean</dt><dd>${(sum / values.length).toPrecision(7)}</dd></div>
      <div class="detail-row"><dt>Signs</dt><dd>${formatNumber(positive, 0)} positive · ${formatNumber(negative, 0)} negative</dd></div>
    </dl>
    <div class="vector-signature-dimensions">${strongest.map((entry) => `<span>D${String(entry.dimension).padStart(4, "0")} · ${entry.value.toPrecision(5)}</span>`).join("")}</div>
  `;
}

async function hydrateVectorSignature(token) {
  const targetId = `embedding-vector-signature-${String(token.index).replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  try {
    const values = await fetchEmbeddingVector(token);
    if (state.selectedToken !== token) return;
    const target = document.getElementById(targetId);
    if (target) {
      target.innerHTML = `<canvas class="vector-heat-strip" width="96" height="64" role="img" aria-label="All 6,144 signed BF16 components arranged in a 64 by 96 grid"></canvas>${vectorSignatureHTML(values)}`;
      const canvas = target.querySelector("canvas");
      const context = canvas?.getContext("2d");
      if (context) {
        const image = context.createImageData(96, 64);
        let maximumMagnitude = 1e-12;
        values.forEach((value) => { maximumMagnitude = Math.max(maximumMagnitude, Math.abs(value)); });
        values.forEach((value, index) => {
          const strength = Math.min(1, Math.sqrt(Math.abs(value) / maximumMagnitude));
          const offset = index * 4;
          if (value < 0) {
            image.data[offset] = Math.round(22 + 50 * strength);
            image.data[offset + 1] = Math.round(45 + 115 * strength);
            image.data[offset + 2] = Math.round(75 + 174 * strength);
          } else {
            image.data[offset] = Math.round(50 + 205 * strength);
            image.data[offset + 1] = Math.round(42 + 92 * strength);
            image.data[offset + 2] = Math.round(48 + 70 * strength);
          }
          image.data[offset + 3] = 255;
        });
        context.putImageData(image, 0, 0);
      }
    }
  } catch (error) {
    const target = document.getElementById(targetId);
    if (target) target.innerHTML = `<div class="method-note">${escapeHTML(error instanceof Error ? error.message : String(error))}</div>`;
  }
}

async function prepareEmbeddingVectorValues(token) {
  const values = await fetchEmbeddingVector(token);
  const rowCount = 64;
  const columnCount = 96;
  const matrix = Array.from({ length: rowCount }, (_, row) =>
    Array.from({ length: columnCount }, (_, column) => values[row * columnCount + column]),
  );
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  values.forEach((value) => {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
    sum += value;
  });
  state.activeValueSample = {
    kind: "embedding-vector",
    tensor: state.embeddingData.tensor?.name ?? "model.llm.embed.weight",
    dtype: state.embeddingData.tensor?.dtype ?? "BF16",
    shape: state.embeddingData.tensor?.shape ?? [201024, EMBEDDING_DIMENSIONS],
    vectorTokenId: Number(token.id),
    vectorTokenDisplay: visibleToken(token),
    leadingIndices: [Number(token.id)],
    rowIndices: Array.from({ length: rowCount }, (_, index) => index),
    columnIndices: Array.from({ length: columnCount }, (_, index) => index),
    values: matrix,
    sampledValues: EMBEDDING_DIMENSIONS,
    minimum,
    maximum,
    mean: sum / values.length,
    method: `All 6,144 exact stored BF16 values from token row ${token.id}, arranged as a 64 × 96 display grid.`,
  };
  state.selectedValue = null;
  state.selectedToken = token;
  return state.activeValueSample;
}

async function openEmbeddingVectorValues(token, { pushHistory = true } = {}) {
  await prepareEmbeddingVectorValues(token);
  await setMode("values", { pushHistory });
}

function updateLayerLabels() {
  if (!state.data) return;
  const layer = state.data.layers[state.selectedLayer];
  dom.layerLabel.textContent = `Layer ${layer.id}`;
  dom.layerKind.textContent = `${layer.attention} attention · ${layer.id < 2 ? "dense MLP" : "sparse MoE"}`;
}

function renderCurrentInspector() {
  if (state.mode === "embeddings") {
    if (state.selectedToken) renderEmbeddingTokenInspector(state.selectedToken);
    else renderEmbeddingOverviewInspector();
  } else if (state.selectedTensor) renderTensorInspector(state.selectedTensor);
  else if (state.selectedExpert) renderExpertInspector(state.selectedExpert);
  else if (state.selectedLayer !== null && state.mode !== "overview") renderLayerInspector(state.selectedLayer);
  else renderOverviewInspector();
}

function embeddingSelectionMetrics() {
  const data = state.embeddingData;
  const tensor = data?.tensor ?? {};
  const tokenCount = data?.tokenCount ?? data?.tokens?.length ?? EXPECTED_TOKEN_COUNT;
  return {
    mappedTokens: tokenCount,
    tokenizerEntries: tokenCount,
    dimensions: Number(data?.dimensions ?? tensor.shape?.[1] ?? EMBEDDING_DIMENSIONS),
    tensorBytes: Number(data?.manifest?.files?.vectors?.bytes ?? tensor.payloadBytes ?? tensor.bytes ?? tokenCount * EMBEDDING_ROW_BYTES),
    paddedExcluded: Number(data?.manifest?.excludedPaddingRows ?? tensor.excludedPaddingRows ?? tensor.unmappedRows ?? data?.unmappedRows ?? 966),
    neighbors: manifestNeighborCount(data?.manifest ?? {}),
  };
}

function renderEmbeddingOverviewInspector() {
  if (!state.embeddingData) return;
  const metrics = embeddingSelectionMetrics();
  const method = state.embeddingData.projection?.method ?? "3D principal-component projection";
  dom.inspectorKicker.textContent = "LEARNED INPUT EMBEDDINGS";
  dom.inspectorTitle.textContent = `${formatNumber(metrics.mappedTokens, 0)} / ${formatNumber(metrics.tokenizerEntries, 0)} token vectors`;
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">The complete tokenizer vocabulary mapped from <code>model.llm.embed.weight</code>. Every point is one real learned input vector; no token entries are sampled away.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>${formatNumber(metrics.mappedTokens, 0)} / ${formatNumber(metrics.tokenizerEntries, 0)}</strong><span>tokens mapped</span></div>
      <div class="metric-card"><strong>${formatNumber(metrics.dimensions, 0)}</strong><span>dimensions each</span></div>
      <div class="metric-card"><strong>${formatNumber(metrics.neighbors, 0)}</strong><span>global neighbours each</span></div>
      <div class="metric-card"><strong>${formatBytes(metrics.tensorBytes)}</strong><span>mapped vector payload</span></div>
    </div>
    <dl class="detail-list">
      <div class="detail-row"><dt>Projection</dt><dd>${escapeHTML(method)}</dd></div>
      <div class="detail-row"><dt>PCA variance</dt><dd>${escapeHTML(projectionVarianceText())}</dd></div>
      <div class="detail-row"><dt>Excluded padding</dt><dd>${formatNumber(metrics.paddedExcluded, 0)} unnamed rows</dd></div>
      <div class="detail-row"><dt>Neighbour search</dt><dd>exhaustive over all ${formatNumber(metrics.tokenizerEntries, 0)}</dd></div>
      <div class="detail-row"><dt>Neighbour metric</dt><dd>exact cosine · original 6,144D</dd></div>
    </dl>
    <div class="section-title">How to read it</div>
    <div class="method-note">The raw learned rows were L2-normalized for comparison and projected into three dimensions. Point proximity is only the projection. Selected spokes are the exact nearest rows found by exhaustive cosine comparison across the complete 200,058-token vocabulary.</div>
    <div class="truth-note" style="margin-top:10px">Tokens are not necessarily whole words. These are static input embeddings, not contextual activations, facts, or the complete knowledge represented after 66 transformer layers.</div>
  `;
  dom.inspectorBody.scrollTop = 0;
  dom.accessibleSummary.textContent = `Complete embedding map with all ${metrics.mappedTokens} tokenizer vectors. Each has ${metrics.dimensions} dimensions and ${metrics.neighbors} exact global cosine neighbours. Search the full vocabulary or select a point.`;
}

function renderSemanticOverviewInspector({ loading = false, error = null } = {}) {
  const metrics = embeddingSelectionMetrics();
  dom.inspectorKicker.textContent = "MEANING MAP";
  if (error) {
    dom.inspectorTitle.textContent = "Meaning map unavailable";
    dom.inspectorBody.innerHTML = `<div class="method-note">${escapeHTML(error instanceof Error ? error.message : String(error))}</div>`;
    return;
  }
  if (loading || !semanticMapData) {
    dom.inspectorTitle.textContent = "Arranging all 200,058 token pieces";
    dom.inspectorBody.innerHTML = '<p class="summary-lede">Building a readable map from every token’s closest learned relationships.</p><div class="method-note">Loading the complete two-dimensional layout…</div>';
    return;
  }
  const manifest = semanticMapData.manifest;
  const clusterCount = Number(manifest.clusterCount ?? semanticMapData.clusterCount);
  const relationshipCount = Number(manifest.method?.graph?.directedInputRelationships ?? metrics.mappedTokens * metrics.neighbors);
  const largestGroups = [...(manifest.clusters ?? [])]
    .sort((a, b) => Number(b.count) - Number(a.count))
    .slice(0, 10)
    .map((cluster) => `<div class="detail-row"><dt>${escapeHTML(String(cluster.label ?? `Region ${Number(cluster.id) + 1}`))}</dt><dd>${formatNumber(Number(cluster.count), 0)} tokens</dd></div>`)
    .join("");
  dom.inspectorTitle.textContent = `${formatNumber(metrics.mappedTokens, 0)} tokens · ${formatNumber(clusterCount, 0)} learned regions`;
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">This view pulls token pieces together when the model learned similar vectors for them. Every one of the 200,058 tokenizer entries is still on the map.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>${formatNumber(metrics.mappedTokens, 0)}</strong><span>token dots</span></div>
      <div class="metric-card"><strong>${formatNumber(clusterCount, 0)}</strong><span>learned regions</span></div>
      <div class="metric-card"><strong>${formatCount(relationshipCount)}</strong><span>closest-token links</span></div>
      <div class="metric-card"><strong>${formatNumber(metrics.dimensions, 0)}</strong><span>values behind each dot</span></div>
    </div>
    <div class="section-title">How to use this map</div>
    <div class="method-note">Scroll to zoom into an island, drag to move around, then click a dot. Its Token close-up will show the exact 32 closest token pieces and their full-vector similarity scores.</div>
    <div class="section-title">Ten largest regions</div>
    <dl class="detail-list">${largestGroups}</dl>
    <div class="truth-note" style="margin-top:14px">Screen distance is a navigational summary, not an exact measurement. The Token close-up is where the exact relationships are shown.</div>
  `;
  dom.inspectorBody.scrollTop = 0;
  dom.accessibleSummary.textContent = `Meaning map of all ${metrics.mappedTokens} tokenizer entries, arranged into ${clusterCount} learned regions from ${relationshipCount} closest-token relationships.`;
}

function renderEmbeddingTokenInspector(token, neighborEntries = null, neighborError = null) {
  const neighbors = Array.isArray(neighborEntries) ? neighborEntries : [];
  const tokenId = Number(token.id ?? token.index);
  const byteRange = [tokenId * EMBEDDING_ROW_BYTES, (tokenId + 1) * EMBEDDING_ROW_BYTES - 1];
  const layoutOffset = embeddingArrayPosition(token) * state.embeddingData.layoutStride;
  const position = [
    state.embeddingData.layout[layoutOffset],
    state.embeddingData.layout[layoutOffset + 1],
    state.embeddingData.layout[layoutOffset + 2],
  ];
  const signatureId = `embedding-vector-signature-${String(token.index).replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  dom.inspectorKicker.textContent = `TOKEN EMBEDDING · ID ${tokenIdentifier(token)}`;
  dom.inspectorTitle.textContent = visibleToken(token);
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">One exact learned input row and its nearest vectors across the complete 200,058-token vocabulary. The bright origin arrow is this 6,144D row projected into three dimensions.</p>
    <dl class="detail-list">
      <div class="detail-row"><dt>Token ID</dt><dd>${escapeHTML(tokenIdentifier(token))}</dd></div>
      <div class="detail-row"><dt>Display</dt><dd>${escapeHTML(visibleToken(token))}</dd></div>
      <div class="detail-row"><dt>Raw piece</dt><dd>${escapeHTML(rawToken(token))}</dd></div>
      <div class="detail-row"><dt>Type</dt><dd>${escapeHTML(token.type ?? token.selectionClass ?? "token")}</dd></div>
      <div class="detail-row"><dt>Script / spacing</dt><dd>${escapeHTML(token.script ?? "None")} · ${token.leadingSpace ? "leading space" : "no leading space"}</dd></div>
      <div class="detail-row"><dt>Stored norm</dt><dd>${Number(token.norm).toPrecision(8)}</dd></div>
      <div class="detail-row"><dt>Full-vector byte range</dt><dd>${formatNumber(byteRange[0], 0)} → ${formatNumber(byteRange[1], 0)}</dd></div>
      <div class="detail-row"><dt>PCA coordinates</dt><dd>${position.slice(0, 3).map((value) => Number(value).toPrecision(5)).join(" · ")}</dd></div>
      <div class="detail-row"><dt>PCA variance</dt><dd>${escapeHTML(projectionVarianceText())}</dd></div>
    </dl>
    <div class="section-title">Relationships</div>
    <div class="method-note">${neighborEntries === null
      ? "Finding the 32 closest token pieces across the complete vocabulary…"
      : neighborError
        ? escapeHTML(neighborError instanceof Error ? neighborError.message : String(neighborError))
        : `Closest token: ${escapeHTML(visibleToken(neighbors[0]?.token ?? token))} · cosine ${Number(neighbors[0]?.record?.cosine ?? 0).toFixed(6)}. All ${neighbors.length} exact neighbours are shown as spokes in the 3D view.`}</div>
    <div class="section-title">All 6,144 signed BF16 components</div>
    <div id="${signatureId}" class="vector-signature"><div class="method-note">Decoding all 6,144 stored values…</div></div>
    <button class="action-button" type="button" data-action="open-token-vector" data-token-index="${escapeHTML(String(token.index))}">Open all 6,144 exact values</button>
    <div class="truth-note" style="margin-top:10px">The neighbour ranking is global and exact in the original 6,144 dimensions. The three-dimensional positions and origin arrow are projections. Tokens are static input pieces, not contextual word meanings.</div>
  `;
  dom.inspectorBody.scrollTop = 0;
  dom.accessibleSummary.textContent = `${visibleToken(token)}, token ID ${tokenIdentifier(token)}, a ${EMBEDDING_DIMENSIONS}-dimensional input embedding with ${neighbors.length || manifestNeighborCount(state.embeddingData.manifest)} global neighbours. The closest is ${neighbors[0] ? `${visibleToken(neighbors[0].token)} at cosine ${neighbors[0].record.cosine}` : "loading"}.`;
  hydrateVectorSignature(token);
}

function renderOverviewInspector() {
  const { architecture, checkpoint, source } = state.data;
  dom.inspectorKicker.textContent = "CHECKPOINT OVERVIEW";
  dom.inspectorTitle.textContent = "Inkling BF16";
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">A spatial index of the published Inkling checkpoint. Every colored block is one named safetensors tensor; the filaments are the model's documented head and expert multiplicities.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>${formatCount(architecture.checkpointParameters)}</strong><span>stored scalars</span></div>
      <div class="metric-card"><strong>${formatBytes(checkpoint.physicalWeightFileBytes)}</strong><span>physical files</span></div>
      <div class="metric-card"><strong>${formatNumber(checkpoint.tensorCount, 0)}</strong><span>tensor keys</span></div>
      <div class="metric-card"><strong>${formatNumber(checkpoint.shardCount, 0)}</strong><span>safetensors files</span></div>
    </div>
    <div class="section-title">Architecture</div>
    <dl class="detail-list">
      <div class="detail-row"><dt>Advertised size</dt><dd>975B total · 41B active</dd></div>
      <div class="detail-row"><dt>Decoder</dt><dd>66 layers · width 6,144</dd></div>
      <div class="detail-row"><dt>Attention</dt><dd>64 Q heads · 8 / 16 KV</dd></div>
      <div class="detail-row"><dt>Experts</dt><dd>256 routed · 6 selected · 2 shared</dd></div>
      <div class="detail-row"><dt>Context</dt><dd>${formatNumber(architecture.contextTokens, 0)} tokens</dd></div>
    </dl>
    <div class="section-title">Stored tensor families</div>
    <dl class="detail-list">${categoryTotalsHTML(checkpoint.categoryParameters)}</dl>
    <div class="section-title">What is measured</div>
    <div class="method-note">${escapeHTML(source.method)} The numerical value view contains exact decoded BF16/F32 values, not generated data.</div>
  `;
  dom.accessibleSummary.textContent = `Inkling checkpoint overview: ${formatCount(architecture.checkpointParameters)} stored scalar values in ${checkpoint.tensorCount} named tensors across ${checkpoint.shardCount} safetensors files.`;
}

function renderLayerInspector(layerId) {
  const layer = state.data.layers[layerId];
  const topBiases = layer.gateBias?.values
    ? [...layer.gateBias.values.keys()].sort((a, b) => layer.gateBias.values[b] - layer.gateBias.values[a]).slice(0, 6)
    : [];
  dom.inspectorKicker.textContent = `DECODER LAYER ${String(layerId).padStart(2, "0")}`;
  dom.inspectorTitle.textContent = layerId < 2 ? "Dense transformer block" : "Sparse expert block";
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">Layer ${layerId} uses ${layer.attention} attention${layer.attention === "local" ? ` with a ${formatNumber(state.data.architecture.slidingWindow, 0)}-token window` : " across the full context"}.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>${formatCount(layer.parameters)}</strong><span>stored scalars</span></div>
      <div class="metric-card"><strong>${formatBytes(layer.bytes)}</strong><span>weight data</span></div>
      <div class="metric-card"><strong>${formatNumber(layer.tensorCount, 0)}</strong><span>named tensors</span></div>
      <div class="metric-card"><strong>${layer.attention}</strong><span>attention scope</span></div>
    </div>
    <div class="section-title">Tensor families</div>
    <dl class="detail-list">${categoryTotalsHTML(layer.categoryParameters)}</dl>
    ${topBiases.length ? `
      <div class="section-title">Highest learned router biases</div>
      <div class="expert-list">${topBiases.map((expert) => `<span class="expert-chip">E${String(expert).padStart(3, "0")} · ${layer.gateBias.values[expert].toPrecision(4)}</span>`).join("")}</div>
      <div class="truth-note" style="margin-top:10px">These are the six highest stored bias values. They are not a real token route; routing also depends on the token state and gate weights.</div>
      <button class="action-button" type="button" data-action="open-experts">Open the 256-expert 3D lattice</button>
    ` : `<div class="method-note">Layers 0 and 1 use dense feed-forward matrices instead of routed experts.</div>`}
  `;
  dom.accessibleSummary.textContent = `Layer ${layerId}: ${layer.attention} attention, ${formatCount(layer.parameters)} stored parameters, ${layer.tensorCount} tensors.`;
}

function renderTensorInspector(tensor) {
  const sample = state.data.matrixSamples[tensor.name];
  const layerLabel = tensor.layer === null ? "Non-decoder" : `Layer ${tensor.layer}`;
  dom.inspectorKicker.textContent = `${CATEGORY_LABELS[tensor.category] ?? "TENSOR"} · ${layerLabel}`.toUpperCase();
  dom.inspectorTitle.textContent = tensor.name;
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">This block is one published tensor record from the pinned Inkling checkpoint.</p>
    <dl class="detail-list">
      <div class="detail-row"><dt>Shape</dt><dd>${escapeHTML(shapeText(tensor.shape))}</dd></div>
      <div class="detail-row"><dt>Data type</dt><dd>${escapeHTML(tensor.dtype)}</dd></div>
      <div class="detail-row"><dt>Scalars</dt><dd>${formatNumber(tensor.parameters, 0)}</dd></div>
      <div class="detail-row"><dt>Weight bytes</dt><dd>${formatBytes(tensor.bytes)}</dd></div>
      <div class="detail-row"><dt>Category</dt><dd>${escapeHTML(CATEGORY_LABELS[tensor.category] ?? tensor.category)}</dd></div>
      <div class="detail-row"><dt>Safetensors file</dt><dd>${escapeHTML(tensor.shard)}</dd></div>
      <div class="detail-row"><dt>Data offsets</dt><dd>${formatNumber(tensor.dataOffsets[0], 0)} → ${formatNumber(tensor.dataOffsets[1], 0)}</dd></div>
    </dl>
    ${tensor.name === "model.llm.embed.weight" ? `
      <div class="section-title">Learned token geometry</div>
      <div class="method-note">Open all 200,058 real 6,144-dimensional tokenizer rows with exhaustive original-space cosine neighbours.</div>
      <button class="action-button" type="button" data-action="open-embeddings">Open the token embedding map</button>
    ` : ""}
    ${sample ? `
      <div class="section-title">Real numerical sample available</div>
      <div class="method-note">${formatNumber(sample.sampledValues, 0)} exact ${sample.dtype} values were byte-range read from this tensor. No values were generated or interpolated.</div>
      <button class="action-button" type="button" data-action="open-values">Open the real values in 3D</button>
    ` : `
      <div class="section-title">Spatial encoding</div>
      <div class="method-note">Block position comes from layer and module role. Block volume is a logarithmic encoding of ${formatNumber(tensor.parameters, 0)} stored scalars.</div>
    `}
  `;
  dom.accessibleSummary.textContent = `${tensor.name}, shape ${shapeText(tensor.shape)}, ${tensor.dtype}, ${formatNumber(tensor.parameters, 0)} scalar values in ${tensor.shard}.`;
}

function renderExpertInspector(detail) {
  const layer = state.data.layers[state.selectedLayer];
  const w13 = state.data.tensors.find((tensor) => tensor.name === `model.llm.layers.${state.selectedLayer}.mlp.experts.w13_weight`);
  const w2 = state.data.tensors.find((tensor) => tensor.name === `model.llm.layers.${state.selectedLayer}.mlp.experts.w2_weight`);
  const paramsPerExpert = (w13?.parameters ?? 0) / 256 + (w2?.parameters ?? 0) / 256;
  dom.inspectorKicker.textContent = `LAYER ${state.selectedLayer} · ROUTED EXPERT`;
  dom.inspectorTitle.textContent = `Expert ${String(detail.expert).padStart(3, "0")}`;
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">One real slice from each of the layer's stacked routed-expert tensors.</p>
    <dl class="detail-list">
      <div class="detail-row"><dt>Expert ID</dt><dd>${detail.expert}</dd></div>
      <div class="detail-row"><dt>Gate bias</dt><dd>${detail.bias.toPrecision(8)}</dd></div>
      <div class="detail-row"><dt>Bias rank set</dt><dd>${detail.topBias ? "highest six" : "not highest six"}</dd></div>
      <div class="detail-row"><dt>Expert scalars</dt><dd>${formatNumber(paramsPerExpert, 0)}</dd></div>
      <div class="detail-row"><dt>w13 slice</dt><dd>[6,144 × 6,144]</dd></div>
      <div class="detail-row"><dt>w2 slice</dt><dd>[6,144 × 3,072]</dd></div>
    </dl>
    <div class="truth-note">Gate bias height is a stored learned value. It is not a routing probability and does not reveal which experts a particular token would select.</div>
    ${state.selectedLayer === 29 && detail.expert === 0 ? `<button class="action-button" type="button" data-action="open-expert-values">Open Expert 000's sampled w13 values</button>` : ""}
  `;
  dom.accessibleSummary.textContent = `Layer ${state.selectedLayer}, expert ${detail.expert}, gate bias ${detail.bias}, ${formatNumber(paramsPerExpert, 0)} expert parameters.`;
}

function renderValueInspector(sample, selectedValue = null) {
  if (!sample) return;
  const tensor = state.data.tensors.find((entry) => entry.name === sample.tensor);
  const isEmbeddingVector = sample.kind === "embedding-vector";
  dom.inspectorKicker.textContent = "EXACT STORED VALUES";
  dom.inspectorTitle.textContent = isEmbeddingVector ? `${sample.vectorTokenDisplay} · embedding row` : sample.tensor;
  const selectedHTML = selectedValue
    ? `
      <div class="section-title">Selected scalar</div>
      <dl class="detail-list">
        <div class="detail-row"><dt>Stored index</dt><dd>${escapeHTML(`[${selectedValue.storedIndex.join(", ")}]`)}</dd></div>
        <div class="detail-row"><dt>Raw ${sample.dtype}</dt><dd>${selectedValue.value.toPrecision(10)}</dd></div>
      </dl>
    `
    : "";
  dom.inspectorBody.innerHTML = `
    <p class="summary-lede">${isEmbeddingVector ? `All 6,144 BF16 values for token row ${sample.vectorTokenId}, arranged as a 64 × 96 display grid without interpolation.` : "Each bar is one exact scalar decoded from the official safetensors payload."} Up/down is sign; height and luminance encode magnitude using a disclosed robust display scale.</p>
    <div class="metric-grid">
      <div class="metric-card"><strong>${formatNumber(sample.sampledValues, 0)}</strong><span>exact values</span></div>
      <div class="metric-card"><strong>${sample.dtype}</strong><span>stored dtype</span></div>
      <div class="metric-card"><strong>${sample.minimum.toPrecision(5)}</strong><span>sample minimum</span></div>
      <div class="metric-card"><strong>${sample.maximum.toPrecision(5)}</strong><span>sample maximum</span></div>
    </div>
    <dl class="detail-list">
      <div class="detail-row"><dt>Full tensor shape</dt><dd>${escapeHTML(shapeText(sample.shape))}</dd></div>
      <div class="detail-row"><dt>Full scalar count</dt><dd>${formatNumber(tensor?.parameters ?? 0, 0)}</dd></div>
      <div class="detail-row"><dt>${isEmbeddingVector ? "Vector" : "Sample"} mean</dt><dd>${sample.mean.toPrecision(7)}</dd></div>
      ${isEmbeddingVector
        ? `<div class="detail-row"><dt>Display layout</dt><dd>64 rows × 96 columns</dd></div><div class="detail-row"><dt>Source row</dt><dd>token ID ${sample.vectorTokenId}</dd></div>`
        : `<div class="detail-row"><dt>Rows sampled</dt><dd>${sample.rowIndices[0]} → ${sample.rowIndices.at(-1)}</dd></div><div class="detail-row"><dt>Columns sampled</dt><dd>${sample.columnIndices[0]} → ${sample.columnIndices.at(-1)}</dd></div>`}
    </dl>
    ${selectedHTML}
    <div class="section-title">Sampling method</div>
    <div class="method-note">${escapeHTML(sample.method)} ${isEmbeddingVector ? "Every dimension is present exactly once." : "Missing cells are not interpolated or displayed as if they were weights."}</div>
    <button class="action-button" type="button" data-action="${isEmbeddingVector ? "back-to-embeddings" : "back-to-tensor"}">${isEmbeddingVector ? "Return to token neighbourhood" : "Return to architecture"}</button>
  `;
  dom.inspectorBody.scrollTop = 0;
  dom.accessibleSummary.textContent = `${isEmbeddingVector ? `${sample.vectorTokenDisplay}, token ${sample.vectorTokenId}` : sample.tensor}: a 3D view of ${sample.sampledValues} exact stored ${sample.dtype} values. Minimum ${sample.minimum}, maximum ${sample.maximum}, mean ${sample.mean}.`;
}

function openInspector() {
  state.inspectorOpen = true;
  dom.inspector.classList.remove("is-closed");
  dom.menuToggleInspector.textContent = "Hide inspector";
  document.querySelector(".workspace").style.gridTemplateColumns = window.innerWidth > 960 ? "minmax(0, 1fr) 370px" : "1fr";
  resizeRenderer();
}

function closeInspector() {
  state.inspectorOpen = false;
  dom.inspector.classList.add("is-closed");
  dom.menuToggleInspector.textContent = "Show inspector";
  document.querySelector(".workspace").style.gridTemplateColumns = "1fr";
  resizeRenderer();
}

function setProjectMenuOpen(open, { restoreFocus = false } = {}) {
  dom.projectMenu.hidden = !open;
  dom.projectMenuButton.setAttribute("aria-expanded", String(open));
  if (open) {
    dom.menuToggleInspector.textContent = state.inspectorOpen ? "Hide inspector" : "Show inspector";
    dom.projectMenuClose.focus();
  } else if (restoreFocus) {
    dom.projectMenuButton.focus();
  }
}

function pointerCoordinates(event) {
  const rect = dom.canvas.getBoundingClientRect();
  return {
    normalized: new THREE.Vector2(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1),
    x: event.clientX,
    y: event.clientY,
  };
}

function pick(event, forClick = false) {
  if (!state.data || !renderer) return null;
  const coordinates = pointerCoordinates(event);
  raycaster.setFromCamera(coordinates.normalized, camera);

  if (state.mode === "overview") {
    const tensorHits = tensorMesh ? raycaster.intersectObject(tensorMesh, false) : [];
    if (tensorHits.length) return { kind: "tensor", id: tensorHits[0].instanceId, point: tensorHits[0].point, ...coordinates };
    if (forClick) {
      const layerHits = layerHitMesh ? raycaster.intersectObject(layerHitMesh, false) : [];
      if (layerHits.length) return { kind: "layer", id: layerHits[0].instanceId, point: layerHits[0].point, ...coordinates };
    }
  } else if (state.mode === "experts") {
    const expertHits = expertMesh ? raycaster.intersectObject(expertMesh, false) : [];
    if (expertHits.length) return { kind: "expert", id: expertHits[0].instanceId, point: expertHits[0].point, ...coordinates };
  } else if (state.mode === "embeddings") {
    const tokenHits = embeddingPoints ? raycaster.intersectObject(embeddingPoints, false) : [];
    if (tokenHits.length) return { kind: "embedding-token", id: tokenHits[0].index, point: tokenHits[0].point, ...coordinates };
  } else if (state.mode === "values") {
    const valueHits = valuesMesh ? raycaster.intersectObject(valuesMesh, false) : [];
    if (valueHits.length) return { kind: "value", id: valueHits[0].instanceId, point: valueHits[0].point, ...coordinates };
  }
  return null;
}

function updateHover(event) {
  if (event.pointerType === "touch") return;
  if (state.mode === "embeddings") {
    state.hovered = null;
    dom.tooltip.hidden = true;
    hoverBox.visible = false;
    return;
  }
  const hit = pick(event, false);
  if (!hit) {
    state.hovered = null;
    dom.tooltip.hidden = true;
    hoverBox.visible = false;
    return;
  }
  state.hovered = hit;
  dom.tooltip.style.left = `${Math.min(window.innerWidth - 360, hit.x + 14)}px`;
  dom.tooltip.style.top = `${Math.min(window.innerHeight - 110, hit.y + 14)}px`;
  dom.tooltip.hidden = false;

  if (hit.kind === "tensor") {
    const tensor = state.data.tensors[hit.id];
    dom.tooltip.innerHTML = `<strong>${escapeHTML(tensor.name)}</strong><span class="tooltip-meta">${escapeHTML(shapeText(tensor.shape))} · ${tensor.dtype} · ${formatCount(tensor.parameters)}</span>`;
    setSelectionBox(tensorLayouts[hit.id], hoverBox);
  } else if (hit.kind === "expert") {
    const expert = expertInstanceData[hit.id];
    dom.tooltip.innerHTML = `<strong>Layer ${state.selectedLayer} · Expert ${String(expert.expert).padStart(3, "0")}</strong><span class="tooltip-meta">gate bias ${expert.bias.toPrecision(7)}${expert.topBias ? " · highest-six bias" : ""}</span>`;
    hoverBox.visible = false;
  } else if (hit.kind === "value") {
    const value = valueInstanceData[hit.id];
    dom.tooltip.innerHTML = `<strong>${value.value.toPrecision(9)}</strong><span class="tooltip-meta">stored index [${value.storedIndex.join(", ")}]</span>`;
    hoverBox.visible = false;
  } else if (hit.kind === "embedding-token") {
    const token = state.embeddingData?.tokens?.[hit.id];
    if (!token) return;
    dom.tooltip.innerHTML = `<strong>${escapeHTML(visibleToken(token))}</strong><span class="tooltip-meta">token ${escapeHTML(tokenIdentifier(token))} · ${escapeHTML(token.type ?? token.selectionClass ?? "token")} · norm ${Number(token.norm).toPrecision(6)}</span>`;
    hoverBox.visible = false;
  }
}

function onPointerDown(event) {
  state.pointerDown = { x: event.clientX, y: event.clientY, time: performance.now() };
}

function onPointerUp(event) {
  if (!state.pointerDown) return;
  const distance = Math.hypot(event.clientX - state.pointerDown.x, event.clientY - state.pointerDown.y);
  const elapsed = performance.now() - state.pointerDown.time;
  state.pointerDown = null;
  if (distance > 6 || elapsed > 650) return;
  const hit = pick(event, true);
  if (!hit) {
    if (state.mode === "overview") {
      state.selectedTensor = null;
      selectionBox.visible = false;
      renderOverviewInspector();
      updateURL(true);
    } else if (state.mode === "embeddings" && state.selectedToken) {
      clearEmbeddingSelection({ pushHistory: true, focus: false });
    }
    return;
  }
  if (hit.kind === "tensor") selectTensor(state.data.tensors[hit.id]);
  if (hit.kind === "layer") selectLayer(hit.id);
  if (hit.kind === "expert") selectExpert(hit.id);
  if (hit.kind === "value") selectValue(hit.id);
  if (hit.kind === "embedding-token") selectEmbeddingToken(state.embeddingData.tokens[hit.id]);
}

function search(query) {
  if (state.mode === "embeddings") {
    searchTokens(query);
    return;
  }
  const normalized = query.trim().toLowerCase();
  if (!normalized) {
    dom.searchResults.hidden = true;
    dom.searchResults.innerHTML = "";
    return;
  }
  const layerMatch = normalized.match(/^(?:layer\s*)?(\d{1,2})$/);
  const results = [];
  if (layerMatch) {
    const id = Number(layerMatch[1]);
    if (id >= 0 && id < state.data.layers.length) results.push({ kind: "layer", layer: state.data.layers[id] });
  }
  const terms = normalized.split(/\s+/).filter(Boolean);
  const tensors = state.data.tensors
    .filter((tensor) => terms.every((term) => tensor.name.toLowerCase().includes(term)))
    .sort((a, b) => b.parameters - a.parameters)
    .slice(0, 12);
  results.push(...tensors.map((tensor) => ({ kind: "tensor", tensor })));
  dom.searchResults.innerHTML = results.length
    ? results
        .map((result, index) => {
          if (result.kind === "layer") {
            return `<button class="search-result" type="button" role="option" data-result="${index}"><span class="shape">LAYER</span><span class="name">Decoder layer ${result.layer.id}</span><span class="size">${formatCount(result.layer.parameters)}</span></button>`;
          }
          return `<button class="search-result" type="button" role="option" data-result="${index}"><span class="shape">${escapeHTML(result.tensor.dtype)}</span><span class="name">${escapeHTML(result.tensor.name)}</span><span class="size">${formatCount(result.tensor.parameters)}</span></button>`;
        })
        .join("")
    : `<div class="method-note" style="margin:10px">No matching tensor keys.</div>`;
  dom.searchResults.hidden = false;
  dom.searchResults.querySelectorAll("[data-result]").forEach((button) => {
    button.addEventListener("click", () => {
      const result = results[Number(button.dataset.result)];
      if (result.kind === "layer") selectLayer(result.layer.id);
      else selectTensor(result.tensor);
      dom.search.value = "";
      dom.searchResults.hidden = true;
    });
  });
}

function tokenSearchScore(token, query) {
  const id = tokenIdentifier(token).toLowerCase();
  const display = visibleToken(token).toLowerCase();
  const displayPlain = display.replaceAll("␠", " ").trim();
  const raw = rawToken(token).toLowerCase();
  const normalizedQuery = query.replaceAll("␠", " ").replace(/^token\s*#?/, "").trim();
  if (id === normalizedQuery) return 0;
  if (displayPlain === normalizedQuery || raw === normalizedQuery || raw === `ġ${normalizedQuery}`) return 1;
  if (displayPlain.startsWith(normalizedQuery) || raw.startsWith(normalizedQuery) || raw.startsWith(`ġ${normalizedQuery}`)) return 2;
  if (displayPlain.includes(normalizedQuery) || raw.includes(normalizedQuery)) return 3;
  return Infinity;
}

function searchTokens(query) {
  const normalized = query.trim().toLowerCase();
  if (!normalized || !state.embeddingData) {
    dom.searchResults.hidden = true;
    dom.searchResults.innerHTML = "";
    return;
  }
  const buckets = [[], [], [], []];
  for (const token of state.embeddingData.tokens) {
    const score = tokenSearchScore(token, normalized);
    if (!Number.isFinite(score) || score > 3) continue;
    if (buckets[score].length < 40) buckets[score].push({ token, score });
  }
  const results = buckets.flat()
    .sort((a, b) => a.score - b.score || visibleToken(a.token).length - visibleToken(b.token).length || Number(a.token.id) - Number(b.token.id))
    .slice(0, 18);
  dom.searchResults.innerHTML = results.length
    ? results.map(({ token }, index) => `
      <button class="search-result token-search-result" type="button" role="option" data-result="${index}">
        <span class="shape">#${escapeHTML(tokenIdentifier(token))}</span>
        <span class="name">${escapeHTML(visibleToken(token))}<small>${escapeHTML(rawToken(token))}</small></span>
        <span class="size">${escapeHTML(token.type ?? token.selectionClass ?? "token")}</span>
      </button>`).join("")
    : `<div class="method-note" style="margin:10px">No matching token in all ${formatNumber(state.embeddingData.tokenCount, 0)} tokenizer entries.</div>`;
  dom.searchResults.hidden = false;
  dom.searchResults.querySelectorAll("[data-result]").forEach((button) => {
    button.addEventListener("click", () => {
      const token = results[Number(button.dataset.result)]?.token;
      if (token) selectEmbeddingToken(token);
      dom.search.value = "";
      dom.searchResults.hidden = true;
    });
  });
}

function updateURL(push = false) {
  const url = new URL(window.location.href);
  url.searchParams.set("view", state.mode);
  url.searchParams.set("layer", String(state.selectedLayer));
  if (state.selectedTensor) url.searchParams.set("tensor", state.selectedTensor.name);
  else url.searchParams.delete("tensor");
  if (state.selectedExpert) url.searchParams.set("expert", String(state.selectedExpert.expert));
  else url.searchParams.delete("expert");
  const activeEmbeddingToken = state.activeValueSample?.kind === "embedding-vector"
    ? String(state.activeValueSample.vectorTokenId)
    : null;
  if (state.mode === "embeddings" && state.selectedToken) url.searchParams.set("token", tokenIdentifier(state.selectedToken));
  else if (state.mode === "values" && activeEmbeddingToken !== null) url.searchParams.set("token", activeEmbeddingToken);
  else url.searchParams.delete("token");
  url.searchParams.delete("map");
  const method = push ? "pushState" : "replaceState";
  window.history[method]({}, "", url);
}

function restoreURL() {
  const params = new URLSearchParams(window.location.search);
  const layerParam = params.get("layer");
  const layer = layerParam === null ? null : Number(layerParam);
  if (layer !== null && Number.isInteger(layer) && layer >= 0 && layer < state.data.layers.length) state.selectedLayer = layer;
  const tensorName = params.get("tensor");
  if (tensorName) state.selectedTensor = state.data.tensors.find((tensor) => tensor.name === tensorName) ?? null;
  const mode = params.get("view");
  if (["overview", "experts", "embeddings", "values"].includes(mode)) state.mode = mode;
  const expertParam = params.get("expert");
  const expertId = expertParam === null ? null : Number(expertParam);
  state.pendingExpertId = expertId !== null && Number.isInteger(expertId) && expertId >= 0 && expertId < 256 ? expertId : null;
  const tokenId = params.get("token");
  state.pendingTokenId = tokenId === null ? null : tokenId;
  state.embeddingView = "all";
  if (state.mode === "values") {
    const activeTokenId = state.activeValueSample?.kind === "embedding-vector"
      ? String(state.activeValueSample.vectorTokenId)
      : null;
    if (tokenId === null || activeTokenId !== tokenId) state.activeValueSample = null;
  } else if (state.activeValueSample?.kind === "embedding-vector") {
    state.activeValueSample = null;
  }
  if (state.mode === "embeddings" && tokenId === null) state.selectedToken = null;
}

function exportPNG() {
  const link = document.createElement("a");
  renderer.render(scene, camera);
  const focus = state.mode === "embeddings" && state.selectedToken ? `token-${tokenIdentifier(state.selectedToken)}` : `layer-${state.selectedLayer}`;
  link.download = `inkling-weight-atlas-${state.mode}-${focus}.png`;
  link.href = renderer.domElement.toDataURL("image/png");
  link.click();
}

function showFallback(message) {
  dom.loading.classList.add("is-hidden");
  dom.fallback.hidden = false;
  dom.fallback.querySelector("p").textContent = message;
  if (state.data) {
    dom.fallbackSummary.textContent = `${formatCount(state.data.architecture.checkpointParameters)} stored scalars · ${state.data.checkpoint.tensorCount} tensors · ${state.data.checkpoint.shardCount} safetensors files`;
  }
}

function bindEvents() {
  dom.canvas.addEventListener("pointermove", updateHover);
  dom.canvas.addEventListener("pointerdown", onPointerDown);
  dom.canvas.addEventListener("pointerup", onPointerUp);
  dom.canvas.addEventListener("pointerleave", () => {
    dom.tooltip.hidden = true;
    hoverBox.visible = false;
  });
  dom.search.addEventListener("input", () => {
    if (embeddingSearchTimer !== null) window.clearTimeout(embeddingSearchTimer);
    if (state.mode !== "embeddings") {
      search(dom.search.value);
      return;
    }
    embeddingSearchTimer = window.setTimeout(() => {
      embeddingSearchTimer = null;
      search(dom.search.value);
    }, 90);
  });
  dom.search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      dom.search.value = "";
      dom.search.blur();
      search("");
    }
  });
  dom.layerSlider.addEventListener("input", () => {
    state.selectedLayer = Number(dom.layerSlider.value);
    updateLayerLabels();
  });
  dom.layerSlider.addEventListener("change", () => selectLayer(Number(dom.layerSlider.value)));
  dom.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  dom.embeddingViewButtons.forEach((button) => button.addEventListener("click", () => setEmbeddingView(button.dataset.embeddingView)));
  dom.labelDensityButtons.forEach((button) => button.addEventListener("click", () => {
    state.embeddingLabelDensity = button.dataset.labelDensity;
    dom.labelDensityButtons.forEach((candidate) => {
      const active = candidate === button;
      candidate.classList.toggle("is-active", active);
      candidate.setAttribute("aria-pressed", String(active));
    });
    if (!state.selectedToken) scheduleEmbeddingLabelRefresh(true);
  }));
  dom.resetView.addEventListener("click", () => {
    if (state.mode === "embeddings") {
      clearEmbeddingSelection({ pushHistory: true, focus: true });
      return;
    }
    state.selectedTensor = null;
    state.selectedExpert = null;
    state.selectedValue = null;
    setMode("overview", { pushHistory: true });
    selectionBox.visible = false;
    renderOverviewInspector();
  });
  dom.exportView.addEventListener("click", exportPNG);
  dom.projectMenuButton.addEventListener("click", () => setProjectMenuOpen(dom.projectMenu.hidden));
  dom.projectMenuClose.addEventListener("click", () => setProjectMenuOpen(false, { restoreFocus: true }));
  dom.copyLocalSetup.addEventListener("click", async () => {
    const previous = dom.copyLocalSetup.textContent;
    try {
      await navigator.clipboard.writeText(LOCAL_SETUP_COMMANDS);
      dom.copyLocalSetup.textContent = "Copied setup commands";
    } catch {
      dom.copyLocalSetup.textContent = "Copy failed — open README";
    }
    window.setTimeout(() => { dom.copyLocalSetup.textContent = previous; }, 1800);
  });
  dom.menuToggleInspector.addEventListener("click", () => {
    if (state.inspectorOpen) closeInspector();
    else openInspector();
    setProjectMenuOpen(false);
  });
  document.addEventListener("pointerdown", (event) => {
    if (!dom.projectMenu.hidden && !dom.projectMenuWrap.contains(event.target)) setProjectMenuOpen(false);
  });
  dom.closeInspector.addEventListener("click", closeInspector);
  dom.inspectorBody.addEventListener("click", (event) => {
    const action = event.target.closest("[data-action]")?.dataset.action;
    const tokenIndex = event.target.closest("[data-token-index]")?.dataset.tokenIndex;
    if (!action && tokenIndex !== undefined) {
      selectEmbeddingToken(embeddingTokenFromReference(tokenIndex));
      return;
    }
    if (action === "open-experts") setMode("experts");
    if (action === "open-values") {
      state.activeValueSample = null;
      setMode("values");
    }
    if (action === "open-embeddings") setMode("embeddings");
    if (action === "open-token-vector") {
      const token = embeddingTokenFromReference(tokenIndex) ?? state.selectedToken;
      if (token) openEmbeddingVectorValues(token).catch((error) => {
        dom.inspectorBody.insertAdjacentHTML("beforeend", `<div class="method-note" style="margin-top:10px">${escapeHTML(error instanceof Error ? error.message : String(error))}</div>`);
      });
    }
    if (action === "open-expert-values") {
      state.activeValueSample = null;
      const tensor = state.data.tensors.find((entry) => entry.name === "model.llm.layers.29.mlp.experts.w13_weight");
      if (tensor) state.selectedTensor = tensor;
      setMode("values");
    }
    if (action === "back-to-tensor") {
      state.activeValueSample = null;
      setMode("overview");
    }
    if (action === "back-to-embeddings") setMode("embeddings");
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== dom.search) {
      event.preventDefault();
      dom.search.focus();
    }
    if (event.key === "Escape") {
      if (!dom.projectMenu.hidden) {
        setProjectMenuOpen(false, { restoreFocus: true });
        return;
      }
      if (state.mode === "embeddings" && state.selectedToken) clearEmbeddingSelection({ pushHistory: true, focus: true });
      else if (state.mode !== "overview") setMode("overview");
      else if (state.selectedTensor) {
        state.selectedTensor = null;
        selectionBox.visible = false;
        renderOverviewInspector();
      }
    }
    if (["ArrowLeft", "ArrowRight"].includes(event.key) && document.activeElement !== dom.search && ["overview", "experts"].includes(state.mode)) {
      event.preventDefault();
      const delta = event.key === "ArrowRight" ? 1 : -1;
      selectLayer(THREE.MathUtils.clamp(state.selectedLayer + delta, 0, 65));
    }
    if (event.key === "Enter" && document.activeElement === dom.layerSlider) selectLayer(Number(dom.layerSlider.value));
  });
  window.addEventListener("popstate", async () => {
    restoreURL();
    updateLayerLabels();
    await setMode(state.mode, { pushHistory: false });
  });
  document.addEventListener("visibilitychange", () => {
    state.renderActive = !document.hidden;
    if (state.renderActive) requestAnimationFrame(animate);
  });
}

function populateStaticUI() {
  const { architecture, checkpoint, source } = state.data;
  dom.statParams.textContent = formatCount(architecture.checkpointParameters);
  dom.statSize.textContent = formatBytes(architecture.checkpointBytes);
  dom.statTensors.textContent = formatNumber(checkpoint.tensorCount, 0);
  dom.statFiles.textContent = formatNumber(checkpoint.shardCount, 0);
  dom.commitLabel.textContent = source.commit.slice(0, 10);
  dom.sourceLink.href = `https://huggingface.co/${source.repository}/tree/${source.commit}`;
  dom.sourceStatus.textContent = `PINNED CHECKPOINT ${source.commit.slice(0, 10)} · ${source.license.toUpperCase()}`;
  updateLayerLabels();
}

function animate(now = performance.now()) {
  if (!state.renderActive || !renderer) return;
  requestAnimationFrame(animate);
  updateCameraTransition(now);
  controls.update();
  updateEmbeddingLabels();
  renderer.render(scene, camera);
  if (!state.sceneReady) {
    state.sceneReady = true;
    document.documentElement.dataset.renderReady = "true";
    dom.loading.classList.add("is-hidden");
  }
}

async function boot() {
  try {
    const response = await fetch("/inkling-weight-map.json");
    if (!response.ok) throw new Error(`Metadata request failed: ${response.status}`);
    state.data = await response.json();
    restoreURL();
    populateStaticUI();
    renderOverviewInspector();
    initThree();
    buildScene(state.data);
    if (state.selectedTensor) {
      const index = findTensorIndex(state.selectedTensor);
      setSelectionBox(tensorLayouts[index]);
      renderTensorInspector(state.selectedTensor);
    }
    if (state.mode === "experts") {
      buildExpertDetail(state.data, state.selectedLayer);
      renderLayerInspector(state.selectedLayer);
      focusExperts(state.selectedLayer);
      if (state.pendingExpertId !== null && expertInstanceData[state.pendingExpertId]) {
        state.selectedExpert = expertInstanceData[state.pendingExpertId];
        renderExpertInspector(state.selectedExpert);
      }
    } else if (state.mode === "embeddings") {
      await setMode("embeddings", { pushHistory: false });
    } else if (state.mode === "values") {
      await setMode("values", { pushHistory: false });
    }
    applyModeVisibility();
    bindEvents();
    if (window.innerWidth <= 960 && ((state.mode === "overview" && !state.selectedTensor) || (state.mode === "embeddings" && !state.selectedToken))) closeInspector();
    updateURL(false);
    requestAnimationFrame(animate);
  } catch (error) {
    console.error(error);
    showFallback(error instanceof Error ? error.message : "The 3D atlas could not start.");
  }
}

boot();
