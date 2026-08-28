import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./styles.css";

const COLORS = {
  background: 0x04080b,
  attention: 0x72a5ff,
  mlp: 0xbb7cff,
  residual: 0x66d9b3,
  active: 0xf0c56a,
  frame: 0x435662,
};

const DEMO_ARCHITECTURE = {
  loaded: false,
  modelId: "EleutherAI/pythia-70m-deduped",
  modelClass: "GPTNeoXForCausalLM",
  modelType: "gpt_neox",
  architectureFamily: "gpt_neox",
  parameterCount: 70_426_624,
  layerCount: 6,
  hiddenSize: 512,
  intermediateSize: 2_048,
  attentionHeads: 8,
  keyValueHeads: 8,
  vocabularySize: 50_304,
  maximumContext: 2_048,
  supportsMlpContributions: true,
  supportsRouterCapture: false,
};

const dom = {
  runtimeDot: document.querySelector("#runtime-dot"),
  runtimeStatus: document.querySelector("#runtime-status"),
  modeButtons: [...document.querySelectorAll("[data-mode]")],
  panels: [...document.querySelectorAll("[data-panel]")],
  loadModel: document.querySelector("#load-model"),
  indexModel: document.querySelector("#index-model"),
  modelCanvas: document.querySelector("#model-canvas"),
  weightWorkspace: document.querySelector("#weight-workspace"),
  weightCanvas: document.querySelector("#weight-canvas"),
  weightTitle: document.querySelector("#weight-title"),
  weightRegion: document.querySelector("#weight-region"),
  weightZoomOut: document.querySelector("#weight-zoom-out"),
  weightReset: document.querySelector("#weight-reset"),
  weightHover: document.querySelector("#weight-hover"),
  weightNote: document.querySelector("#weight-note"),
  anatomyHud: document.querySelector("#anatomy-hud"),
  hudModel: document.querySelector("#hud-model"),
  hudLayers: document.querySelector("#hud-layers"),
  hudParameters: document.querySelector("#hud-parameters"),
  hudSelected: document.querySelector("#hud-selected"),
  sceneLegend: document.querySelector("#scene-legend"),
  sceneInstructions: document.querySelector("#scene-instructions"),
  timeline: document.querySelector("#timeline"),
  timelineLabel: document.querySelector("#timeline-label"),
  timelineSlider: document.querySelector("#timeline-slider"),
  tokenStrip: document.querySelector("#token-strip"),
  stageLoading: document.querySelector("#stage-loading"),
  loadingTitle: document.querySelector("#loading-title"),
  loadingDetail: document.querySelector("#loading-detail"),
  anatomyTitle: document.querySelector("#anatomy-title"),
  anatomyContent: document.querySelector("#anatomy-content"),
  tensorSearch: document.querySelector("#tensor-search"),
  tensorResults: document.querySelector("#tensor-results"),
  tensorDetails: document.querySelector("#tensor-details"),
  scalarDetails: document.querySelector("#scalar-details"),
  traceTitle: document.querySelector("#trace-title"),
  stepSelect: document.querySelector("#step-select"),
  layerSelect: document.querySelector("#layer-select"),
  traceSummary: document.querySelector("#trace-summary"),
  attentionCanvas: document.querySelector("#attention-canvas"),
  neuronTable: document.querySelector("#neuron-table"),
  logitLens: document.querySelector("#logit-lens"),
  outputIndex: document.querySelector("#output-index"),
  inspectContribution: document.querySelector("#inspect-contribution"),
  contributionResults: document.querySelector("#contribution-results"),
  promptForm: document.querySelector("#prompt-form"),
  promptInput: document.querySelector("#prompt-input"),
  promptStatus: document.querySelector("#prompt-status"),
  maxTokens: document.querySelector("#max-tokens"),
  runTrace: document.querySelector("#run-trace"),
};

const state = {
  mode: "anatomy",
  online: false,
  health: null,
  architecture: { ...DEMO_ARCHITECTURE },
  selectedLayer: 0,
  tensors: [],
  selectedTensor: null,
  tile: null,
  tileRects: [],
  region: null,
  regionStack: [],
  selectedScalar: null,
  traceSession: null,
  traceSteps: [],
  traceComplete: null,
  selectedStep: -1,
  socket: null,
  searchTimer: null,
};

let renderer;
let scene;
let camera;
let controls;
let raycaster;
let pointer;
let anatomyGroup;
let selectionBox;
let layerObjects = [];
let resizeObserver;
let renderActive = true;

function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value, digits = 2) {
  return new Intl.NumberFormat("en", { maximumFractionDigits: digits }).format(Number(value));
}

function formatCount(value) {
  const number = Number(value || 0);
  if (number >= 1e12) return `${(number / 1e12).toFixed(3)}T`;
  if (number >= 1e9) return `${(number / 1e9).toFixed(3)}B`;
  if (number >= 1e6) return `${(number / 1e6).toFixed(2)}M`;
  if (number >= 1e3) return `${(number / 1e3).toFixed(2)}K`;
  return formatNumber(number, 0);
}

function formatBytes(value) {
  const number = Number(value || 0);
  if (number >= 1e12) return `${(number / 1e12).toFixed(3)} TB`;
  if (number >= 1e9) return `${(number / 1e9).toFixed(2)} GB`;
  if (number >= 1e6) return `${(number / 1e6).toFixed(2)} MB`;
  if (number >= 1e3) return `${(number / 1e3).toFixed(2)} KB`;
  return `${number} B`;
}

function tokenLabel(token) {
  const value = String(token?.text || token?.piece || token?.id || "");
  return value.replaceAll("\n", "↵").replaceAll("\t", "⇥") || "∅";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  if (!response.ok) {
    const message = payload?.detail || payload?.message || `${response.status} ${response.statusText}`;
    throw new Error(message);
  }
  return payload;
}

function setRuntimeStatus(text, status = "idle") {
  dom.runtimeStatus.textContent = text;
  dom.runtimeDot.dataset.status = status;
}

function setLoading(visible, title = "Preparing model", detail = "Reading checkpoint metadata") {
  dom.stageLoading.hidden = !visible;
  dom.loadingTitle.textContent = title;
  dom.loadingDetail.textContent = detail;
}

function detailRows(entries) {
  return `<dl class="detail-list">${entries.map(([label, value]) => `
    <div class="detail-row"><dt>${escapeHTML(label)}</dt><dd>${value}</dd></div>`).join("")}</dl>`;
}

function metricCards(entries) {
  return `<div class="metric-grid">${entries.map(([value, label]) => `
    <div class="metric-card"><strong>${value}</strong><span>${escapeHTML(label)}</span></div>`).join("")}</div>`;
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
  if (!hasWebGL()) {
    dom.anatomyContent.innerHTML = `<p class="truth-note">WebGL is unavailable. The exact weight and trace inspectors remain usable.</p>`;
    return;
  }
  renderer = new THREE.WebGLRenderer({
    canvas: dom.modelCanvas,
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setClearColor(COLORS.background, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(COLORS.background, 0.015);
  camera = new THREE.PerspectiveCamera(38, 1, 0.1, 500);
  camera.position.set(26, 18, 31);

  controls = new OrbitControls(camera, dom.modelCanvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.075;
  controls.minDistance = 8;
  controls.maxDistance = 100;
  controls.target.set(0, 0, 0);
  controls.update();

  scene.add(new THREE.AmbientLight(0xb8d8e8, 1.2));
  const key = new THREE.DirectionalLight(0xeaf7ff, 2.5);
  key.position.set(18, 28, 24);
  const rim = new THREE.DirectionalLight(0x667cff, 1.4);
  rim.position.set(-20, 4, -18);
  scene.add(key, rim);

  anatomyGroup = new THREE.Group();
  scene.add(anatomyGroup);

  const grid = new THREE.GridHelper(90, 45, 0x25343c, 0x10191e);
  grid.position.y = -5.4;
  grid.material.transparent = true;
  grid.material.opacity = 0.36;
  scene.add(grid);

  selectionBox = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.95, depthTest: false }),
  );
  selectionBox.visible = false;
  selectionBox.renderOrder = 20;
  scene.add(selectionBox);

  raycaster = new THREE.Raycaster();
  pointer = new THREE.Vector2(2, 2);
  dom.modelCanvas.addEventListener("pointerup", selectFromScene);
  resizeObserver = new ResizeObserver(resizeVisuals);
  resizeObserver.observe(dom.modelCanvas.parentElement);
  resizeVisuals();
  requestAnimationFrame(animate);
}

function disposeObject(object) {
  object.traverse?.((child) => {
    child.geometry?.dispose?.();
    if (Array.isArray(child.material)) child.material.forEach((material) => material.dispose?.());
    else child.material?.dispose?.();
  });
}

function moduleMaterial(color) {
  return new THREE.MeshStandardMaterial({
    color,
    emissive: color,
    emissiveIntensity: 0.05,
    roughness: 0.42,
    metalness: 0.12,
    transparent: true,
    opacity: 0.9,
  });
}

function buildAnatomy(architecture = state.architecture) {
  if (!anatomyGroup) return;
  while (anatomyGroup.children.length) disposeObject(anatomyGroup.children.pop());
  layerObjects = [];
  const layerCount = Math.max(1, Number(architecture.layerCount || 6));
  const spacing = Math.min(5.2, Math.max(2.5, 30 / Math.max(6, layerCount)));
  const startX = -((layerCount - 1) * spacing) / 2;
  const residualPoints = [];

  for (let layer = 0; layer < layerCount; layer += 1) {
    const x = startX + layer * spacing;
    const group = new THREE.Group();
    group.position.x = x;
    anatomyGroup.add(group);

    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(2.2, 9.2, 5.5)),
      new THREE.LineBasicMaterial({ color: COLORS.frame, transparent: true, opacity: 0.42 }),
    );
    frame.userData = { kind: "layer", layer };
    group.add(frame);

    const residual = new THREE.Mesh(new THREE.BoxGeometry(0.7, 7.9, 0.7), moduleMaterial(COLORS.residual));
    residual.userData = { kind: "module", module: "residual", layer, baseColor: COLORS.residual };
    group.add(residual);

    const attention = new THREE.Mesh(new THREE.BoxGeometry(1.65, 2.35, 3.9), moduleMaterial(COLORS.attention));
    attention.position.y = 2.45;
    attention.userData = { kind: "module", module: "attention", layer, baseColor: COLORS.attention };
    group.add(attention);

    const mlp = new THREE.Mesh(new THREE.BoxGeometry(1.65, 2.7, 4.35), moduleMaterial(COLORS.mlp));
    mlp.position.y = -2.45;
    mlp.userData = { kind: "module", module: "mlp", layer, baseColor: COLORS.mlp };
    group.add(mlp);

    const labelCanvas = document.createElement("canvas");
    labelCanvas.width = 128;
    labelCanvas.height = 64;
    const context = labelCanvas.getContext("2d");
    context.fillStyle = "rgba(4,8,11,0.86)";
    context.fillRect(0, 0, 128, 64);
    context.fillStyle = "#e7f1f5";
    context.font = "24px ui-monospace, monospace";
    context.textAlign = "center";
    context.fillText(`L${layer}`, 64, 40);
    const texture = new THREE.CanvasTexture(labelCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const label = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
    label.scale.set(2.2, 1.1, 1);
    label.position.set(0, 5.25, 0);
    group.add(label);

    residualPoints.push(new THREE.Vector3(x, 0, 0));
    layerObjects.push({ layer, group, frame, residual, attention, mlp });
  }

  const residualGeometry = new THREE.BufferGeometry().setFromPoints(residualPoints);
  anatomyGroup.add(new THREE.Line(
    residualGeometry,
    new THREE.LineBasicMaterial({ color: COLORS.residual, transparent: true, opacity: 0.58 }),
  ));

  selectLayer(Math.min(state.selectedLayer, layerCount - 1), { focus: false });
  const extent = Math.max(16, layerCount * spacing * 0.6);
  controls.maxDistance = Math.max(80, extent * 4);
}

function resizeVisuals() {
  if (renderer) {
    const rect = dom.modelCanvas.parentElement.getBoundingClientRect();
    renderer.setSize(Math.max(1, rect.width), Math.max(1, rect.height), false);
    camera.aspect = Math.max(1, rect.width) / Math.max(1, rect.height);
    camera.updateProjectionMatrix();
  }
  drawWeightTile();
  renderAttention();
}

function animate() {
  if (!renderActive || !renderer) return;
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function selectFromScene(event) {
  if (!renderer || state.mode === "weights") return;
  const rect = dom.modelCanvas.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const candidates = layerObjects.flatMap((entry) => [entry.residual, entry.attention, entry.mlp, entry.frame]);
  const hit = raycaster.intersectObjects(candidates, false)[0];
  if (hit?.object?.userData?.layer !== undefined) {
    selectLayer(Number(hit.object.userData.layer));
  }
}

function selectLayer(layer, { focus = true } = {}) {
  const maximum = Math.max(0, Number(state.architecture.layerCount || 1) - 1);
  state.selectedLayer = Math.min(maximum, Math.max(0, Number(layer)));
  dom.layerSelect.value = String(state.selectedLayer);
  dom.hudSelected.textContent = `Layer ${state.selectedLayer}`;
  const record = layerObjects[state.selectedLayer];
  if (record && selectionBox) {
    const world = new THREE.Vector3();
    record.group.getWorldPosition(world);
    selectionBox.position.copy(world);
    selectionBox.scale.set(2.45, 9.5, 5.8);
    selectionBox.visible = true;
    if (focus) {
      const direction = camera.position.clone().sub(controls.target).normalize();
      controls.target.copy(world);
      camera.position.copy(world).add(direction.multiplyScalar(20));
    }
  }
  renderAnatomyInspector();
  renderTrace();
}

function resetAnatomyActivity() {
  layerObjects.forEach((entry) => {
    for (const module of [entry.residual, entry.attention, entry.mlp]) {
      module.material.color.setHex(module.userData.baseColor);
      module.material.emissive.setHex(module.userData.baseColor);
      module.material.emissiveIntensity = 0.05;
      module.material.opacity = 0.84;
    }
  });
}

function applyTraceActivity(layerRecords) {
  resetAnatomyActivity();
  if (!Array.isArray(layerRecords) || !layerRecords.length) return;
  const maximumDelta = Math.max(...layerRecords.map((layer) => Number(layer.deltaNorm || 0)), 1e-8);
  const maximumMlp = Math.max(...layerRecords.map((layer) => Number(layer.mlp?.norm || 0)), 1e-8);
  layerRecords.forEach((layer) => {
    const record = layerObjects[Number(layer.layer)];
    if (!record) return;
    const residualStrength = Math.min(1, Number(layer.deltaNorm || 0) / maximumDelta);
    const mlpStrength = Math.min(1, Number(layer.mlp?.norm || 0) / maximumMlp);
    const attentionStrength = Math.min(1, Number(layer.attention?.edges?.[0]?.weight || 0));
    const active = new THREE.Color(COLORS.active);
    for (const [mesh, strength] of [
      [record.residual, residualStrength],
      [record.attention, attentionStrength],
      [record.mlp, mlpStrength],
    ]) {
      const base = new THREE.Color(mesh.userData.baseColor);
      mesh.material.color.copy(base.clone().lerp(active, strength * 0.58));
      mesh.material.emissive.copy(active);
      mesh.material.emissiveIntensity = 0.08 + strength * 1.35;
      mesh.material.opacity = 0.58 + strength * 0.42;
    }
  });
}

function setMode(mode) {
  if (!["anatomy", "weights", "trace"].includes(mode)) return;
  state.mode = mode;
  dom.modeButtons.forEach((button) => button.classList.toggle("is-active", button.dataset.mode === mode));
  dom.panels.forEach((panel) => panel.classList.toggle("is-active", panel.dataset.panel === mode));
  const weights = mode === "weights";
  dom.modelCanvas.hidden = weights;
  dom.weightWorkspace.hidden = !weights;
  dom.anatomyHud.hidden = weights;
  dom.sceneLegend.hidden = weights;
  dom.sceneInstructions.hidden = weights;
  dom.timeline.hidden = mode !== "trace" || state.traceSteps.length === 0;
  if (weights) {
    if (!state.tensors.length && state.online) searchTensors(dom.tensorSearch.value);
    requestAnimationFrame(drawWeightTile);
  }
  if (mode === "trace") renderTrace();
}

function populateArchitecture(architecture) {
  state.architecture = { ...DEMO_ARCHITECTURE, ...architecture };
  const modelName = String(state.architecture.modelId || "Open model").split("/").at(-1);
  dom.hudModel.textContent = modelName;
  dom.hudLayers.textContent = formatNumber(state.architecture.layerCount, 0);
  dom.hudParameters.textContent = formatCount(state.architecture.parameterCount);
  dom.outputIndex.max = String(Math.max(0, Number(state.architecture.hiddenSize || 1) - 1));
  dom.layerSelect.innerHTML = Array.from({ length: Number(state.architecture.layerCount || 1) }, (_, layer) => (
    `<option value="${layer}">Layer ${layer}</option>`
  )).join("");
  state.selectedLayer = Math.min(state.selectedLayer, Number(state.architecture.layerCount || 1) - 1);
  buildAnatomy(state.architecture);
  renderAnatomyInspector();
}

function renderAnatomyInspector() {
  const architecture = state.architecture;
  const selected = state.traceSteps[state.selectedStep]?.layers?.[state.selectedLayer];
  dom.anatomyTitle.textContent = selected ? `Layer ${state.selectedLayer} during generation` : "Transformer overview";
  const live = selected ? `
    <div class="section-heading"><strong>Selected live layer</strong><span>generation step ${state.selectedStep}</span></div>
    ${metricCards([
      [Number(selected.norm).toPrecision(6), "residual norm"],
      [Number(selected.deltaNorm).toPrecision(6), "residual change"],
      [Number(selected.mlp?.norm || 0).toPrecision(6), "MLP activation norm"],
      [Number(selected.attention?.meanEntropy || 0).toPrecision(5), "mean attention entropy"],
    ])}
  ` : "";
  dom.anatomyContent.innerHTML = `
    <p class="summary-lede">A stable navigational anatomy of the loaded decoder. Every layer and module corresponds to real model structure; spatial placement is schematic.</p>
    ${metricCards([
      [formatCount(architecture.parameterCount), "parameters"],
      [formatNumber(architecture.layerCount, 0), "decoder layers"],
      [formatNumber(architecture.hiddenSize, 0), "residual width"],
      [formatNumber(architecture.attentionHeads, 0), "attention heads"],
    ])}
    ${detailRows([
      ["Model", escapeHTML(architecture.modelId)],
      ["Architecture", escapeHTML(architecture.modelClass || architecture.modelType)],
      ["MLP width", formatNumber(architecture.intermediateSize, 0)],
      ["Vocabulary", formatNumber(architecture.vocabularySize, 0)],
      ["Maximum context", formatNumber(architecture.maximumContext, 0)],
      ["Execution", `${escapeHTML(architecture.device || "not loaded")} · ${escapeHTML(architecture.dtype || "configured on load")}`],
    ])}
    ${live}
    <div class="truth-note">Brightness in Live trace mode is driven by measured residual change, MLP activation magnitude, and selected attention weights. It is not a claim that one layer is “thinking harder.”</div>
  `;
}

async function checkHealth() {
  try {
    const health = await api("/api/health");
    state.online = true;
    state.health = health;
    setRuntimeStatus(
      health.runtime.loaded
        ? `${health.runtime.modelId} loaded on ${health.runtime.device}`
        : `Backend ready · ${health.runtime.modelId} configured`,
      health.runtime.loaded ? "live" : "ready",
    );
    if (health.runtime.loaded) {
      const architecture = await api("/api/model");
      populateArchitecture(architecture);
      await searchTensors("");
    }
  } catch (error) {
    state.online = false;
    setRuntimeStatus("Backend offline · showing anatomy shell", "offline");
    dom.promptStatus.textContent = "Start the FastAPI backend to record real inference and checkpoint values.";
    console.warn(error);
  }
}

async function loadModel() {
  if (!state.online) {
    await checkHealth();
    if (!state.online) return;
  }
  setLoading(true, "Loading open model", "Downloading or opening the checkpoint, then constructing the exact tensor index");
  dom.loadModel.disabled = true;
  try {
    const result = await api("/api/model/load", {
      method: "POST",
      body: JSON.stringify({}),
    });
    populateArchitecture(result.architecture);
    setRuntimeStatus(`${result.architecture.modelId} loaded on ${result.architecture.device}`, "live");
    dom.promptStatus.textContent = "Real forward-pass instrumentation is ready.";
    await searchTensors("");
  } catch (error) {
    setRuntimeStatus(`Model load failed: ${error.message}`, "offline");
  } finally {
    setLoading(false);
    dom.loadModel.disabled = false;
  }
}

async function indexModel() {
  if (!state.online) return;
  setLoading(true, "Indexing checkpoint", "Reading safetensors headers without loading all numerical values");
  dom.indexModel.disabled = true;
  try {
    const manifest = await api("/api/checkpoint/index", { method: "POST", body: "{}" });
    setRuntimeStatus(`${formatNumber(manifest.tensorCount, 0)} tensors indexed exactly`, "ready");
    await searchTensors(dom.tensorSearch.value);
    setMode("weights");
  } catch (error) {
    setRuntimeStatus(`Checkpoint indexing failed: ${error.message}`, "offline");
  } finally {
    setLoading(false);
    dom.indexModel.disabled = false;
  }
}

async function searchTensors(query) {
  if (!state.online) {
    dom.tensorResults.innerHTML = `<p class="empty-state">Start the backend to search exact checkpoint tensors.</p>`;
    return;
  }
  try {
    const result = await api(`/api/tensors?q=${encodeURIComponent(query)}&limit=120`);
    state.tensors = result.items;
    renderTensorResults();
  } catch (error) {
    dom.tensorResults.innerHTML = `<p class="empty-state">${escapeHTML(error.message)}</p>`;
  }
}

function renderTensorResults() {
  dom.tensorResults.innerHTML = state.tensors.length
    ? state.tensors.map((tensor, index) => `
      <button class="tensor-result ${state.selectedTensor?.name === tensor.name ? "is-selected" : ""}" type="button" data-tensor-index="${index}">
        <span class="tensor-shape">${escapeHTML(`[${tensor.shape.join(" × ")}]`)}</span>
        <strong>${escapeHTML(tensor.name)}</strong>
        <small>${escapeHTML(tensor.dtype)} · ${formatCount(tensor.scalarCount)} values</small>
      </button>
    `).join("")
    : `<p class="empty-state">No matching tensors.</p>`;
  dom.tensorResults.querySelectorAll("[data-tensor-index]").forEach((button) => {
    button.addEventListener("click", () => selectTensor(state.tensors[Number(button.dataset.tensorIndex)]));
  });
}

async function selectTensor(tensor) {
  state.selectedTensor = tensor;
  state.selectedScalar = null;
  state.region = {
    rowStart: 0,
    rowCount: Number(tensor.matrixRows),
    columnStart: 0,
    columnCount: Number(tensor.matrixColumns),
  };
  state.regionStack = [];
  dom.scalarDetails.innerHTML = "";
  renderTensorResults();
  renderTensorDetails();
  setMode("weights");
  await loadTile();
}

function renderTensorDetails() {
  const tensor = state.selectedTensor;
  if (!tensor) {
    dom.tensorDetails.innerHTML = `<p class="empty-state">Select a tensor to enter the weight atlas.</p>`;
    return;
  }
  dom.tensorDetails.innerHTML = `
    <div class="section-heading"><strong>Selected tensor</strong><span>real safetensors record</span></div>
    ${detailRows([
      ["Name", `<code>${escapeHTML(tensor.name)}</code>`],
      ["Shape", escapeHTML(`[${tensor.shape.join(" × ")}]`)],
      ["Data type", escapeHTML(tensor.dtype)],
      ["Scalars", formatNumber(tensor.scalarCount, 0)],
      ["Weight bytes", formatBytes(tensor.tensorBytes)],
      ["Source", `<code>${escapeHTML(tensor.sourceFile)}</code>`],
      ["Payload start", formatNumber(tensor.absoluteDataStart, 0)],
      ["Atlas matrix", `${formatNumber(tensor.matrixRows, 0)} × ${formatNumber(tensor.matrixColumns, 0)}`],
    ])}
    <div class="truth-note">${escapeHTML(tensor.matrixRule)}</div>
  `;
}

async function loadTile() {
  if (!state.selectedTensor || !state.region) return;
  const region = state.region;
  setLoading(true, "Reading exact weight region", `${formatNumber(region.rowCount * region.columnCount, 0)} source values`);
  try {
    const query = new URLSearchParams({
      name: state.selectedTensor.name,
      rowStart: String(region.rowStart),
      rowCount: String(region.rowCount),
      columnStart: String(region.columnStart),
      columnCount: String(region.columnCount),
      targetRows: "64",
      targetColumns: "64",
    });
    state.tile = await api(`/api/tile?${query}`);
    dom.weightTitle.textContent = state.selectedTensor.name;
    dom.weightRegion.textContent = `rows ${formatNumber(region.rowStart, 0)}–${formatNumber(region.rowStart + region.rowCount - 1, 0)} · columns ${formatNumber(region.columnStart, 0)}–${formatNumber(region.columnStart + region.columnCount - 1, 0)}`;
    dom.weightZoomOut.disabled = state.regionStack.length === 0;
    dom.weightReset.disabled = false;
    dom.weightNote.textContent = state.tile.sourceValues === state.tile.outputRows * state.tile.outputColumns
      ? "Maximum zoom: every displayed heatmap cell is one exact checkpoint scalar."
      : `${formatNumber(state.tile.sourceValues, 0)} exact source values summarized into ${formatNumber(state.tile.outputRows * state.tile.outputColumns, 0)} navigational cells.`;
    drawWeightTile();
  } catch (error) {
    state.tile = null;
    dom.weightNote.textContent = error.message;
  } finally {
    setLoading(false);
  }
}

function robustHeatScale(cells) {
  const values = cells.flatMap((row) => row.map((cell) => Math.max(Math.abs(cell.minimum), Math.abs(cell.maximum))));
  values.sort((a, b) => a - b);
  return Math.max(values[Math.floor(values.length * 0.97)] || values.at(-1) || 1, 1e-12);
}

function heatColor(value, scale) {
  const normalized = Math.max(-1, Math.min(1, value / scale));
  const strength = Math.sqrt(Math.abs(normalized));
  if (normalized < 0) {
    return `rgb(${Math.round(14 + 42 * strength)}, ${Math.round(43 + 94 * strength)}, ${Math.round(70 + 185 * strength)})`;
  }
  return `rgb(${Math.round(35 + 220 * strength)}, ${Math.round(34 + 78 * (1 - strength))}, ${Math.round(45 + 56 * (1 - strength))})`;
}

function drawWeightTile() {
  const canvas = dom.weightCanvas;
  const tile = state.tile;
  if (!canvas || !tile || dom.weightWorkspace.hidden) return;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  const rows = tile.outputRows;
  const columns = tile.outputColumns;
  const cellWidth = rect.width / columns;
  const cellHeight = rect.height / rows;
  const scale = robustHeatScale(tile.cells);
  state.tileRects = [];
  for (let row = 0; row < rows; row += 1) {
    for (let column = 0; column < columns; column += 1) {
      const cell = tile.cells[row][column];
      const x = column * cellWidth;
      const y = row * cellHeight;
      context.fillStyle = heatColor(cell.mean, scale);
      context.fillRect(x, y, Math.ceil(cellWidth + 0.4), Math.ceil(cellHeight + 0.4));
      if (cell.isScalar && cellWidth >= 8 && cellHeight >= 8) {
        context.strokeStyle = "rgba(255,255,255,0.13)";
        context.strokeRect(x + 0.5, y + 0.5, cellWidth - 1, cellHeight - 1);
      }
      state.tileRects.push({ x, y, width: cellWidth, height: cellHeight, cell });
    }
  }
}

function weightCellAt(event) {
  const rect = dom.weightCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return state.tileRects.find((entry) => x >= entry.x && x < entry.x + entry.width && y >= entry.y && y < entry.y + entry.height) || null;
}

function showWeightHover(event) {
  const entry = weightCellAt(event);
  if (!entry) {
    dom.weightHover.hidden = true;
    return;
  }
  const cell = entry.cell;
  dom.weightHover.hidden = false;
  dom.weightHover.style.left = `${event.clientX - dom.weightCanvas.getBoundingClientRect().left + 12}px`;
  dom.weightHover.style.top = `${event.clientY - dom.weightCanvas.getBoundingClientRect().top + 12}px`;
  dom.weightHover.innerHTML = `
    <strong>${cell.isScalar ? "Exact scalar" : `${formatNumber(cell.count, 0)} values`}</strong>
    <span>rows ${cell.rowStart}–${cell.rowStart + cell.rowCount - 1}</span>
    <span>columns ${cell.columnStart}–${cell.columnStart + cell.columnCount - 1}</span>
    <span>mean ${Number(cell.mean).toPrecision(6)}</span>
    <span>min ${Number(cell.minimum).toPrecision(6)} · max ${Number(cell.maximum).toPrecision(6)}</span>
  `;
}

async function activateWeightCell(event) {
  const entry = weightCellAt(event);
  if (!entry || !state.selectedTensor) return;
  const cell = entry.cell;
  if (cell.isScalar) {
    await loadScalar(cell.rowStart, cell.columnStart);
    return;
  }
  state.regionStack.push({ ...state.region });
  state.region = {
    rowStart: cell.rowStart,
    rowCount: cell.rowCount,
    columnStart: cell.columnStart,
    columnCount: cell.columnCount,
  };
  await loadTile();
}

async function loadScalar(matrixRow, matrixColumn) {
  if (!state.selectedTensor) return;
  try {
    const query = new URLSearchParams({
      name: state.selectedTensor.name,
      matrixRow: String(matrixRow),
      matrixColumn: String(matrixColumn),
    });
    state.selectedScalar = await api(`/api/scalar?${query}`);
    const scalar = state.selectedScalar;
    dom.scalarDetails.innerHTML = `
      <div class="section-heading"><strong>One stored weight</strong><span>literal checkpoint bytes</span></div>
      ${metricCards([
        [Number(scalar.value).toPrecision(10), "decoded value"],
        [escapeHTML(scalar.dtype), "stored dtype"],
      ])}
      ${detailRows([
        ["Tensor indices", `<code>[${scalar.indices.join(", ")}]</code>`],
        ["Matrix coordinate", `<code>[${scalar.matrixRow}, ${scalar.matrixColumn}]</code>`],
        ["Flat index", formatNumber(scalar.flatIndex, 0)],
        ["Raw hexadecimal", `<code>${escapeHTML(scalar.rawHex)}</code>`],
        ["Raw bits", `<code class="bit-string">${escapeHTML(scalar.rawBits)}</code>`],
        ["Source file", `<code>${escapeHTML(scalar.sourceFile)}</code>`],
        ["Absolute byte offset", formatNumber(scalar.absoluteByteOffset, 0)],
        ["Byte range", `${formatNumber(scalar.byteRange[0], 0)}–${formatNumber(scalar.byteRange[1], 0)}`],
      ])}
      <div class="truth-note">This is the exact scalar stored at this checkpoint address, decoded from the displayed raw bytes.</div>
    `;
    dom.scalarDetails.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    dom.scalarDetails.innerHTML = `<p class="empty-state">${escapeHTML(error.message)}</p>`;
  }
}

function currentContextTokens() {
  if (!state.traceSession || state.selectedStep < 0) return [];
  return [
    ...(state.traceSession.inputTokens || []),
    ...state.traceSteps.slice(0, state.selectedStep).map((step) => step.selectedToken),
  ];
}

function selectTraceStep(index) {
  if (!state.traceSteps.length) return;
  state.selectedStep = Math.max(0, Math.min(state.traceSteps.length - 1, Number(index)));
  dom.timelineSlider.value = String(state.selectedStep);
  dom.stepSelect.value = String(state.selectedStep);
  const step = state.traceSteps[state.selectedStep];
  applyTraceActivity(step.layers);
  renderTimeline();
  renderTrace();
  renderAnatomyInspector();
}

function renderTimeline() {
  const steps = state.traceSteps;
  dom.timeline.hidden = state.mode !== "trace" || steps.length === 0;
  dom.timelineSlider.max = String(Math.max(0, steps.length - 1));
  dom.timelineSlider.disabled = steps.length === 0;
  const current = steps[state.selectedStep];
  dom.timelineLabel.textContent = current
    ? `Step ${state.selectedStep} · generated ${tokenLabel(current.selectedToken)}`
    : "No trace recorded";
  dom.tokenStrip.innerHTML = steps.map((step, index) => `
    <button type="button" class="token-chip ${index === state.selectedStep ? "is-selected" : ""}" data-step="${index}">
      <span>${index}</span><strong>${escapeHTML(tokenLabel(step.selectedToken))}</strong>
    </button>
  `).join("");
  dom.tokenStrip.querySelectorAll("[data-step]").forEach((button) => {
    button.addEventListener("click", () => selectTraceStep(Number(button.dataset.step)));
  });
  dom.stepSelect.innerHTML = steps.map((step, index) => `
    <option value="${index}">${index} · ${escapeHTML(tokenLabel(step.selectedToken))}</option>
  `).join("");
  dom.stepSelect.disabled = steps.length === 0;
}

function renderRankTable(entries, rendererFunction) {
  if (!entries?.length) return `<p class="empty-state">No measured values for this selection.</p>`;
  return entries.map(rendererFunction).join("");
}

function renderTrace() {
  const step = state.traceSteps[state.selectedStep];
  const layer = step?.layers?.[state.selectedLayer];
  if (!step || !layer) {
    dom.traceTitle.textContent = "Awaiting a prompt";
    dom.traceSummary.innerHTML = `<p class="summary-lede">Run a prompt to record the model's real forward pass, then scrub generated tokens and inspect any layer.</p>`;
    dom.neuronTable.innerHTML = `<p class="empty-state">No trace recorded.</p>`;
    dom.logitLens.innerHTML = `<p class="empty-state">No trace recorded.</p>`;
    dom.inspectContribution.disabled = true;
    renderAttention();
    return;
  }
  dom.traceTitle.textContent = `${tokenLabel(step.selectedToken)} · layer ${state.selectedLayer}`;
  const predictions = step.topPredictions?.slice(0, 5) || [];
  dom.traceSummary.innerHTML = `
    ${metricCards([
      [escapeHTML(tokenLabel(step.selectedToken)), "selected token"],
      [`${Number(step.latencyMilliseconds).toFixed(1)} ms`, "forward pass"],
      [Number(layer.norm).toPrecision(6), "residual norm"],
      [Number(layer.deltaNorm).toPrecision(6), "layer change"],
    ])}
    <div class="section-heading"><strong>Final candidates</strong><span>full-vocabulary probabilities</span></div>
    <div class="prediction-list">${predictions.map((prediction) => `
      <div><strong>${escapeHTML(tokenLabel(prediction))}</strong><span>${(Number(prediction.probability) * 100).toFixed(3)}%</span><i style="--prediction:${Math.max(0.01, Number(prediction.probability))}"></i></div>
    `).join("")}</div>
  `;
  const neurons = layer.mlp?.topDimensions || [];
  dom.neuronTable.innerHTML = renderRankTable(neurons, (entry, rank) => `
    <div class="rank-row"><span>${String(rank + 1).padStart(2, "0")}</span><strong>Neuron ${formatNumber(entry.index, 0)}</strong><em>${Number(entry.value).toPrecision(7)}</em></div>
  `);
  dom.logitLens.innerHTML = renderRankTable(layer.logitLens || [], (entry, rank) => `
    <div class="rank-row"><span>${String(rank + 1).padStart(2, "0")}</span><strong>${escapeHTML(tokenLabel(entry))}</strong><em>${(Number(entry.probability) * 100).toFixed(3)}%</em></div>
  `);
  dom.inspectContribution.disabled = !state.traceSession?.sessionId || !state.architecture.supportsMlpContributions;
  renderAttention();
}

function renderAttention() {
  const canvas = dom.attentionCanvas;
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, rect.width, rect.height);
  context.fillStyle = "rgba(255,255,255,0.025)";
  context.fillRect(0, 0, rect.width, rect.height);

  const step = state.traceSteps[state.selectedStep];
  const layer = step?.layers?.[state.selectedLayer];
  const tokens = currentContextTokens();
  const edges = layer?.attention?.edges || [];
  if (!tokens.length || !edges.length) {
    context.fillStyle = "rgba(210,226,233,0.55)";
    context.font = "12px ui-monospace, monospace";
    context.fillText("No attention matrix captured for this selection.", 18, 30);
    return;
  }

  const maximumVisible = 24;
  const firstPosition = Math.max(0, tokens.length - maximumVisible);
  const visible = tokens.slice(firstPosition);
  const left = 18;
  const right = rect.width - 18;
  const baseline = rect.height - 35;
  const spacing = visible.length <= 1 ? 0 : (right - left) / (visible.length - 1);
  const xFor = (absolutePosition) => left + (absolutePosition - firstPosition) * spacing;

  context.font = "10px ui-monospace, monospace";
  context.textAlign = "center";
  visible.forEach((token, index) => {
    const x = left + index * spacing;
    context.fillStyle = "rgba(233,243,247,0.82)";
    const label = tokenLabel(token);
    context.fillText(label.length > 8 ? `${label.slice(0, 7)}…` : label, x, baseline + 18);
    context.fillStyle = "rgba(102,217,179,0.8)";
    context.fillRect(x - 1.5, baseline - 1.5, 3, 3);
  });

  const fromPosition = tokens.length - 1;
  const fromX = xFor(fromPosition);
  edges.forEach((edge, index) => {
    if (edge.toPosition < firstPosition || edge.toPosition >= tokens.length) return;
    const targetX = xFor(edge.toPosition);
    const distance = Math.abs(fromX - targetX);
    const height = Math.max(24, Math.min(rect.height - 55, 28 + distance * 0.34 + index * 4));
    context.beginPath();
    context.moveTo(fromX, baseline);
    context.quadraticCurveTo((fromX + targetX) / 2, baseline - height, targetX, baseline);
    context.strokeStyle = `rgba(114,165,255,${Math.max(0.18, Math.min(0.94, Number(edge.weight)))})`;
    context.lineWidth = 1 + Number(edge.weight) * 3;
    context.stroke();
    context.fillStyle = "rgba(240,197,106,0.92)";
    context.fillText(`H${edge.head}`, (fromX + targetX) / 2, baseline - height - 4);
  });
}

async function runTrace(event) {
  event.preventDefault();
  if (!state.online) {
    await checkHealth();
    if (!state.online) return;
  }
  if (state.socket) state.socket.close();
  state.traceSession = null;
  state.traceSteps = [];
  state.traceComplete = null;
  state.selectedStep = -1;
  dom.contributionResults.innerHTML = "";
  resetAnatomyActivity();
  setMode("trace");
  setLoading(true, "Starting functional scan", "Loading the model and preparing forward-pass hooks");
  dom.runTrace.disabled = true;
  dom.promptStatus.textContent = "Recording measured internal tensors…";

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${protocol}://${window.location.host}/api/trace/ws`);
  state.socket = socket;
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({
      prompt: dom.promptInput.value,
      maxNewTokens: Number(dom.maxTokens.value),
      temperature: 0,
      topP: 0.95,
      captureAttention: true,
      captureLogitLens: true,
      seed: 0,
    }));
  });
  socket.addEventListener("message", (message) => {
    const payload = JSON.parse(message.data);
    if (payload.type === "model_loading") {
      setLoading(true, "Loading open model", `${payload.modelId} · ${payload.device}`);
      return;
    }
    if (payload.type === "model_ready") return;
    if (payload.type === "session_start") {
      state.traceSession = payload;
      populateArchitecture(payload.architecture);
      setLoading(false);
      setRuntimeStatus(`${payload.architecture.modelId} · live trace`, "live");
      return;
    }
    if (payload.type === "token_step") {
      state.traceSteps.push(payload);
      selectTraceStep(state.traceSteps.length - 1);
      dom.promptStatus.textContent = `Generated ${state.traceSteps.length} token${state.traceSteps.length === 1 ? "" : "s"}; recording continues.`;
      return;
    }
    if (payload.type === "complete") {
      state.traceComplete = payload;
      dom.promptStatus.textContent = `Trace complete: ${payload.generatedText || "generation ended"}`;
      setRuntimeStatus("Trace complete · values retained for inspection", "ready");
      return;
    }
    if (payload.type === "error") {
      dom.promptStatus.textContent = `${payload.error}: ${payload.message}`;
      setRuntimeStatus(`Trace failed: ${payload.message}`, "offline");
      setLoading(false);
    }
  });
  socket.addEventListener("close", () => {
    state.socket = null;
    dom.runTrace.disabled = false;
    setLoading(false);
    renderTimeline();
  });
  socket.addEventListener("error", () => {
    dom.promptStatus.textContent = "The trace WebSocket could not connect to the backend.";
    setRuntimeStatus("Trace connection failed", "offline");
  });
}

async function inspectContribution() {
  const step = state.traceSteps[state.selectedStep];
  if (!step || !state.traceSession) return;
  dom.inspectContribution.disabled = true;
  dom.contributionResults.innerHTML = `<p class="empty-state">Computing exact products…</p>`;
  try {
    const result = await api("/api/contribution", {
      method: "POST",
      body: JSON.stringify({
        sessionId: state.traceSession.sessionId,
        generationStep: state.selectedStep,
        layer: state.selectedLayer,
        outputIndex: Number(dom.outputIndex.value),
        topK: 32,
      }),
    });
    dom.contributionResults.innerHTML = `
      ${detailRows([
        ["Weight tensor", `<code>${escapeHTML(result.weightTensor)}</code>`],
        ["Weighted sum", Number(result.weightedSum).toPrecision(9)],
        ["Bias", Number(result.bias).toPrecision(9)],
        ["Reconstructed output", Number(result.reconstructedLinearOutput).toPrecision(9)],
      ])}
      <div class="contribution-list">${result.topContributions.map((entry, rank) => `
        <article>
          <span>${String(rank + 1).padStart(2, "0")}</span>
          <div><strong>Input ${formatNumber(entry.inputIndex, 0)}</strong><small>activation ${Number(entry.activation).toPrecision(7)}</small></div>
          <div><strong>${Number(entry.contribution).toPrecision(8)}</strong><small>${Number(entry.weight).toPrecision(7)} × activation</small></div>
          ${entry.checkpointAddress ? `<code title="${escapeHTML(entry.checkpointAddress.rawBits)}">byte ${formatNumber(entry.checkpointAddress.absoluteByteOffset, 0)}</code>` : ""}
        </article>
      `).join("")}</div>
      <div class="truth-note">Each row is one measured multiplication for this MLP linear output. The ranking is by absolute local contribution, not by complete causal responsibility for the generated token.</div>
    `;
  } catch (error) {
    dom.contributionResults.innerHTML = `<p class="empty-state">${escapeHTML(error.message)}</p>`;
  } finally {
    dom.inspectContribution.disabled = false;
  }
}

function bindEvents() {
  dom.modeButtons.forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  dom.loadModel.addEventListener("click", loadModel);
  dom.indexModel.addEventListener("click", indexModel);
  dom.tensorSearch.addEventListener("input", () => {
    window.clearTimeout(state.searchTimer);
    state.searchTimer = window.setTimeout(() => searchTensors(dom.tensorSearch.value), 120);
  });
  dom.weightCanvas.addEventListener("pointermove", showWeightHover);
  dom.weightCanvas.addEventListener("pointerleave", () => { dom.weightHover.hidden = true; });
  dom.weightCanvas.addEventListener("click", activateWeightCell);
  dom.weightZoomOut.addEventListener("click", async () => {
    const previous = state.regionStack.pop();
    if (!previous) return;
    state.region = previous;
    await loadTile();
  });
  dom.weightReset.addEventListener("click", async () => {
    if (!state.selectedTensor) return;
    state.regionStack = [];
    state.region = {
      rowStart: 0,
      rowCount: Number(state.selectedTensor.matrixRows),
      columnStart: 0,
      columnCount: Number(state.selectedTensor.matrixColumns),
    };
    await loadTile();
  });
  dom.promptForm.addEventListener("submit", runTrace);
  dom.timelineSlider.addEventListener("input", () => selectTraceStep(Number(dom.timelineSlider.value)));
  dom.stepSelect.addEventListener("change", () => selectTraceStep(Number(dom.stepSelect.value)));
  dom.layerSelect.addEventListener("change", () => selectLayer(Number(dom.layerSelect.value), { focus: false }));
  dom.inspectContribution.addEventListener("click", inspectContribution);
  window.addEventListener("keydown", (event) => {
    if (event.key === "/" && document.activeElement !== dom.tensorSearch && state.mode === "weights") {
      event.preventDefault();
      dom.tensorSearch.focus();
    }
    if (["ArrowLeft", "ArrowRight"].includes(event.key) && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
      const delta = event.key === "ArrowRight" ? 1 : -1;
      selectLayer(state.selectedLayer + delta);
    }
  });
  document.addEventListener("visibilitychange", () => {
    renderActive = !document.hidden;
    if (renderActive && renderer) requestAnimationFrame(animate);
  });
}

async function boot() {
  initThree();
  populateArchitecture(DEMO_ARCHITECTURE);
  renderTensorDetails();
  renderTrace();
  bindEvents();
  setMode("anatomy");
  await checkHealth();
}

boot();
