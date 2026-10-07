/**
 * UI wiring for the watermarking playground. Inference runs in worker.js.
 *
 * One worker per load attempt: switching models or failing a load terminates the
 * worker and spawns a fresh one (see worker.js for why). A WebGPU failure is retried
 * once on WASM; an out-of-memory failure ends with an explanation + diagnostics.
 */
import {
  APP_VERSION,
  MODELS,
  WEBGPU_ONLY,
  DTYPES,
  modelInfo,
  sizeMB,
  estimatedLoadMB,
  formatMB,
  classifyError,
  collectEnvironment,
  probeWasmHeapMB,
  createLog,
} from "./models.js?v=7";
import { expectedTournamentMean } from "./watermark.js?v=10";

const $ = (id) => document.getElementById(id);

const els = {
  model: $("model"),
  loadCachedModel: $("load-cached-model"),
  downloadModel: $("download-model"),
  cachedModelList: $("cached-model-list"),
  activeModel: $("active-model"),
  statusText: $("status-text"),
  statusNote: $("status-note"),
  backendBadge: $("backend-badge"),
  progress: $("progress"),
  progressFill: $("progress-fill"),
  diag: $("diagnostics"),
  diagText: $("diag-text"),
  copyDiag: $("copy-diag"),
  prompt: $("prompt"),
  maxTokens: $("maxTokens"),
  maxInput: $("maxInput"),
  temperature: $("temperature"),
  topK: $("topK"),
  topP: $("topP"),
  seed: $("seed"),
  gamma: $("gamma"),
  delta: $("delta"),
  ctxWidth: $("ctxWidth"),
  layers: $("layers"),
  wmKey: $("wmKey"),
  detectKey: $("detectKey"),
  redWords: $("redWords"),
  redWordsField: $("redwords-field"),
  legendHi: $("legend-hi"),
  legendLo: $("legend-lo"),
  statFirstLabel: $("stat-first-label"),
  statScheme: $("stat-scheme"),
  generateBtn: $("generate-btn"),
  stopBtn: $("stop-btn"),
  detectBtn: $("detect-btn"),
  seedShown: $("seed-shown"),
  output: $("output"),
  placeholder: $("output-placeholder"),
  generateTip: $("generate-tip"),
  detectTip: $("detect-tip"),
  promptEcho: $("prompt-echo"),
  tokens: $("tokens"),
  legend: $("legend"),
  colorText: $("color-text"),
  verdict: $("verdict"),
  verdictLabel: $("verdict-label"),
  verdictConf: $("verdict-conf"),
  meterFill: $("meter-fill"),
  statGreen: $("stat-green"),
  statZ: $("stat-z"),
  statP: $("stat-p"),
  verdictNote: $("verdict-note"),
};

let modelReady = false;
let modelLoading = false;
let downloadingModel = false;
let generating = false;
let haveGeneration = false;
let outputDirty = false;
let lastPaint = null;
let streamSpan = null;

/* ── URL overrides (debugging / perf comparison) ── */
const params = new URLSearchParams(location.search);
// ?device=wasm|webgpu pins the backend (no automatic WASM retry when pinned)
const forcedDevice = ["wasm", "webgpu"].includes(params.get("device")) ? params.get("device") : null;
// ?dtype=q4|q4f16|q8|fp16|fp32 pins the quantisation
const forcedDtype = DTYPES.includes(params.get("dtype")) ? params.get("dtype") : null;
// ?debug=1 always shows the diagnostics panel and mirrors worker messages to the console
const DEBUG = params.has("debug");

const LOAD_PHASES = new Set(["starting", "backend", "download", "compile"]);
const PHASE_TEXT = {
  starting: "запуска среды",
  backend: "выбора устройства",
  download: "загрузки",
  compile: "компиляции модели",
  generate: "генерации",
  detect: "проверки",
};

/* ── worker lifecycle ── */

const log = createLog(80);
let active = null; // current attempt: { id, worker, modelId, device, dtype, retriedWasm, phase, … }
let attemptSeq = 0;
let lastBackend = null; // last { device, dtype, f16, note, adapter, … } reported by a worker
let crashReloads = 0;

function spawnWorker(meta) {
  const id = ++attemptSeq;
  const runtime = modelInfo(meta.modelId).runtime;
  const script = runtime === "v4"
    ? "worker-v4.js?v=18"
    : runtime === "v4next"
      ? "worker-next.js?v=4"
      : "worker.js?v=17";
  const worker = new Worker(script, { type: "module" });
  // Ignore events from a worker we already retired (a message can be queued before terminate()).
  worker.onmessage = (e) => {
    if (active?.id === id) handleMessage(e.data);
  };
  worker.onerror = (e) => {
    if (active?.id === id) handleWorkerError(e);
  };
  active = { id, worker, phase: "starting", startedAt: performance.now(), bytes: null, backend: null, hello: null, workerLog: null, ...meta };
  log.push("spawn", { id, modelId: meta.modelId, device: meta.device ?? "auto", dtype: meta.dtype ?? "auto", retriedWasm: !!meta.retriedWasm });
}

function retireWorker() {
  if (!active) return;
  log.push("retire", { id: active.id });
  active.worker.onmessage = null;
  active.worker.onerror = null;
  active.worker.terminate();
  active = null;
}

const send = (msg) => active?.worker.postMessage(msg);

/** Start a load attempt in a fresh worker. meta: { modelId, device, dtype, retriedWasm, keepOutput, firstFailure } */
function startLoad(meta) {
  retireWorker();
  modelReady = false;
  modelLoading = true;
  els.activeModel.textContent = `Загружается: ${modelInfo(meta.modelId).name}`;
  generating = false;
  streamSpan = null;
  haveGeneration = false; // the last generation lived in the old worker
  els.backendBadge.hidden = true;
  els.progress.hidden = true;
  els.progressFill.style.width = "0%";
  if (!meta.keepOutput) resetOutput();
  if (!meta.retriedWasm) {
    showNote("");
    hideDiagnostics();
  }
  updateButtons();
  setStatus(meta.cacheOnly ? "Загружаю модель из кэша…" : "Запускаю среду…");
  spawnWorker(meta);
  send({ type: "load", modelId: meta.modelId, device: meta.device ?? null, dtype: meta.dtype ?? null, cacheOnly: !!meta.cacheOnly });
}

/* ── helpers ── */

const clamp = (v, lo, hi, fallback) => {
  const n = parseFloat(v);
  return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : fallback;
};

const shorten = (s, n = 160) => {
  s = String(s ?? "");
  return s.length > n ? `${s.slice(0, n)}…` : s;
};

const getMode = () => document.querySelector('input[name="wmMode"]:checked').value;
const getDetectMode = () => document.querySelector('input[name="detectMode"]:checked')?.value ?? getMode();
const MODE_NAME = { none: "Нет", hard: "Жёсткий список", soft: "Мягкий список", tournament: "Турнир SynthID" };

function readParams() {
  return {
    prompt: els.prompt.value,
    promptStyle: "chat",
    mode: getMode(),
    maxNewTokens: Math.round(clamp(els.maxTokens.value, 1, 8192, 4096)),
    maxInputTokens: Math.round(clamp(els.maxInput.value, 128, 32768, 16384)),
    temperature: clamp(els.temperature.value, 0, 2, 1),
    topK: Math.round(clamp(els.topK.value, 0, 1000, 200)),
    topP: clamp(els.topP.value, 0.05, 1, 0.9),
    seed: 42,
    gamma: clamp(els.gamma.value, 0.05, 0.95, 0.5),
    delta: clamp(els.delta.value, 0, 15, 4),
    h: Math.round(clamp(els.ctxWidth.value, 1, 8, 1)),
    m: Math.round(clamp(els.layers.value, 1, 30, 15)),
    key: els.wmKey.value,
    redWords: els.redWords.value,
  };
}

/** level: "info" | "warn" | "error" */
function setStatus(text, level = "info") {
  els.statusText.textContent = text;
  els.statusText.classList.toggle("status-error", level === "error");
  els.statusText.classList.toggle("status-warn", level === "warn");
}

/** Secondary line under the status (advice, retry explanation). */
function showNote(text) {
  els.statusNote.textContent = text;
  els.statusNote.hidden = !text;
}

function showBackendBadge({ device, dtype }) {
  els.backendBadge.textContent = device === "webgpu" ? `WebGPU · ${dtype}` : `WASM · ${dtype} (медленнее)`;
  const b = lastBackend;
  els.backendBadge.title =
    (device === "webgpu"
      ? "Вычисления модели выполняются на видеокарте через WebGPU."
      : "Вычисления модели выполняются на процессоре через WebAssembly в одном потоке.") +
    " " + [b?.adapter?.description || b?.adapter?.vendor, b?.note].filter(Boolean).join(" — ");
  els.backendBadge.hidden = false;
}

function updateButtons() {
  els.generateBtn.disabled = !modelReady || generating || modelLoading || downloadingModel;
  const hasPaste = outputDirty;
  els.detectBtn.disabled = generating || modelLoading || downloadingModel || !modelReady || (!haveGeneration && !hasPaste);
  els.stopBtn.hidden = !generating;
  els.model.disabled = generating || modelLoading || downloadingModel;
  els.loadCachedModel.disabled = generating || modelLoading || downloadingModel || !cachedModels.has(els.model.value);
  els.downloadModel.disabled = generating || modelLoading || downloadingModel;
}

/** Rewrite the dropdown labels with the sizes for the active backend/dtype. */
function refreshModelOptions(backend) {
  for (const opt of els.model.options) {
    const m = modelInfo(opt.value);
    const dtype = m.webgpuOnly
      ? (backend?.device === "webgpu" ? backend.dtype : "q4f16")
      : backend?.dtype;
    const size = dtype ? sizeMB(opt.value, dtype) : null;
    const parts = [];
    if (size) parts.push(`~${(size / 1000).toFixed(1).replace(".", ",")} ГБ`);
    if (m.webgpuOnly) parts.push("только WebGPU");
    opt.textContent = parts.length ? `${m.name} (${parts.join(", ")})` : m.name;
    opt.disabled = !!m.webgpuOnly && !!backend && backend.device !== "webgpu";
  }
}

const MODEL_CACHE_NAME = "transformers-cache";
let cachedModels = new Map();

async function refreshCachedModels() {
  if (!("caches" in window)) {
    els.cachedModelList.textContent = "Этот браузер не поддерживает просмотр кэша моделей.";
    return;
  }
  try {
    const cache = await caches.open(MODEL_CACHE_NAME);
    const requests = await cache.keys();
    const next = new Map();
    for (const entry of MODELS) {
      const repoPath = `/${entry.id}/resolve/`;
      const weights = requests.filter((request) => {
        const path = new URL(request.url).pathname;
        return path.includes(repoPath) && /[^/]+\.onnx$/i.test(path);
      });
      if (!weights.length) continue;
      const variants = new Set();
      for (const request of weights) {
        const match = new URL(request.url).pathname.match(/_(q4f16|q8|q4|fp16|fp32|int8|uint8)\.onnx$/i);
        if (match) variants.add(match[1].toLowerCase());
      }
