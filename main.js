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
      const variantList = [...variants];
      next.set(entry.id, {
        count: weights.length,
        variants: variantList,
        size: variantList.length ? entry.sizesMB[variantList[0]] : null,
      });
    }
    cachedModels = next;
    renderCachedModels();
    refreshModelOptions(lastBackend);
    updateButtons();
  } catch (error) {
    els.cachedModelList.textContent = `Не удалось прочитать кэш: ${error.message ?? error}`;
  }
}

async function getDownloadDtype(modelId) {
  const info = modelInfo(modelId);
  if (info.runtime === "v4" || info.runtime === "v4next") return "q4f16";
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter && !adapter.isFallbackAdapter) {
      return adapter.features.has("shader-f16") ? "q4f16" : "q4";
    }
  } catch {}
  return info.webgpuOnly ? "q4f16" : "q8";
}

function isWantedModelFile(path) {
  if (!path.includes("/")) {
    return path !== "README.md" && path !== ".gitattributes" && !path.startsWith(".");
  }
  if (!path.startsWith("onnx/")) return false;
  return /^(?:model|decoder_model_merged|embed_tokens)_q4f16\.onnx(?:_data(?:_\d+)?)?$/i.test(path.slice("onnx/".length));
}

async function downloadSelectedModel() {
  const modelId = els.model.value;
  const info = modelInfo(modelId);
  downloadingModel = true;
  els.progress.hidden = false;
  els.progressFill.style.width = "0%";
  showNote("");
  updateButtons();
  try {
    const dtype = await getDownloadDtype(modelId);
    const api = `https://huggingface.co/api/models/${modelId}/tree/main?recursive=true&expand=false`;
    setStatus(`Получаю список файлов ${info.name}…`);
    const listing = await fetch(api);
    if (!listing.ok) throw new Error(`Список файлов Hugging Face вернул HTTP ${listing.status}.`);
    const allFiles = await listing.json();
    const files = allFiles
      .filter((file) => file.type === "file" && isWantedModelFile(file.path))
      .filter((file) => !file.path.startsWith("onnx/") || new RegExp(`_${dtype}\\.onnx(?:_data(?:_\\d+)?)?$`, "i").test(file.path.slice(5)))
      .sort((a, b) => a.path.localeCompare(b.path));
    if (!files.some((file) => /^onnx\/(?:model|decoder_model_merged)_/.test(file.path))) {
      throw new Error(`В репозитории ${info.name} не нашёл ONNX-веса для ${dtype}.`);
    }

    const cache = await caches.open(MODEL_CACHE_NAME);
    const totalBytes = files.reduce((sum, file) => sum + (file.size || 0), 0);
    let finishedBytes = 0;
    let saved = 0;
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const url = `https://huggingface.co/${modelId}/resolve/main/${file.path}`;
      const request = new Request(url, { mode: "cors" });
      const cached = await cache.match(request);
      if (!cached) {
        const pct = totalBytes ? Math.floor(100 * finishedBytes / totalBytes) : Math.floor(100 * i / files.length);
        els.progressFill.style.width = `${pct}%`;
        setStatus(`Скачиваю ${info.name}: ${file.path} (${i + 1}/${files.length})`);
        const response = await fetch(request);
        if (!response.ok) throw new Error(`${file.path}: HTTP ${response.status}.`);
        const body = response.clone().body;
        const cacheWrite = cache.put(request, response);
        if (body) {
          const reader = body.getReader();
          let fileLoaded = 0;
          let lastPaint = 0;
          for (;;) {
            const chunk = await reader.read();
            if (chunk.done) break;
            fileLoaded += chunk.value.byteLength;
            const now = performance.now();
            if (now - lastPaint > 250) {
              const fraction = totalBytes ? (finishedBytes + fileLoaded) / totalBytes : (i + fileLoaded / Math.max(1, file.size || fileLoaded)) / files.length;
              els.progressFill.style.width = `${Math.min(100, Math.floor(100 * fraction))}%`;
              setStatus(`Скачиваю ${info.name}: ${file.path} (${formatMB(fileLoaded / 1e6)} / ${formatMB((file.size || 0) / 1e6)})`);
              lastPaint = now;
            }
          }
        }
        await cacheWrite;
        saved++;
      }
      finishedBytes += file.size || 0;
      const pct = totalBytes ? Math.floor(100 * finishedBytes / totalBytes) : Math.floor(100 * (i + 1) / files.length);
      els.progressFill.style.width = `${Math.min(100, pct)}%`;
    }
    log.push("cache-download", { modelId, dtype, files: files.length, saved, bytes: totalBytes });
    await refreshCachedModels();
    setStatus(`${info.name} скачана. Загружаю модель в память…`);
    els.progressFill.style.width = "100%";
    crashReloads = 0;
    startLoad({ modelId, device: forcedDevice, dtype: forcedDtype, cacheOnly: true });
  } catch (error) {
    setStatus(`Не удалось скачать ${info.name}: ${error.message ?? error}`, "error");
    showNote("Уже скачанные файлы остались в кэше. Можно повторить скачивание: готовые файлы будут пропущены.");
    await refreshCachedModels();
  } finally {
    downloadingModel = false;
    els.progress.hidden = true;
    updateButtons();
  }
}

function renderCachedModels() {
  els.cachedModelList.replaceChildren();
  if (!cachedModels.size) {
    els.cachedModelList.textContent = "Скачанных моделей пока нет.";
    return;
  }
  for (const entry of MODELS) {
    const cached = cachedModels.get(entry.id);
    if (!cached) continue;
    const row = document.createElement("div");
    row.className = "cached-model-row";
    const label = document.createElement("span");
    const variant = cached.variants.length ? cached.variants.join(", ").toUpperCase() : "ONNX";
    const size = cached.size ? ` · ~${(cached.size / 1000).toFixed(1).replace(".", ",")} ГБ` : "";
    label.textContent = `${entry.name} (${variant}${size})`;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn secondary small";
    remove.textContent = "Удалить";
    remove.addEventListener("click", () => deleteCachedModel(entry));
    row.append(label, remove);
    els.cachedModelList.append(row);
  }
}

async function deleteCachedModel(entry) {
  try {
    if (active?.modelId === entry.id) {
      retireWorker();
      modelReady = false;
      modelLoading = false;
      els.activeModel.textContent = "Модель не загружена";
      generating = false;
      haveGeneration = false;
      els.backendBadge.hidden = true;
      updateButtons();
      setStatus("Модель выгружена перед удалением кэша.");
    }
    const cache = await caches.open(MODEL_CACHE_NAME);
    const repoPath = `/${entry.id}/resolve/`;
    let deleted = 0;
    for (const request of await cache.keys()) {
      if (new URL(request.url).pathname.includes(repoPath) && await cache.delete(request)) deleted++;
    }
    log.push("cache-delete", { modelId: entry.id, requests: deleted });
    await refreshCachedModels();
    setStatus(`Кэш модели ${entry.name} удалён.`);
  } catch (error) {
    setStatus(`Не удалось удалить кэш: ${error.message ?? error}`, "error");
  }
}

/** Which parameters each mode exposes (ids of .param wrappers / fields). */
const VISIBLE_PARAMS = {
  none: ["maxTokens", "maxInput", "temperature", "topK", "topP", "seed"],
  hard: ["maxTokens", "maxInput", "temperature", "topK", "topP", "seed", "wmSection", "gamma", "ctxWidth", "wmKey", "redWords"],
  soft: ["maxTokens", "maxInput", "temperature", "topK", "topP", "seed", "wmSection", "gamma", "delta", "ctxWidth", "wmKey", "redWords"],
  tournament: ["maxTokens", "maxInput", "temperature", "topK", "topP", "seed", "wmSection", "layers", "ctxWidth", "wmKey", "redWords"],
};

const MODE_HINT = {
  none: "Создаёт ответ без водяного знака. Модель выбирает продолжения по своим вероятностям и настройкам генерации.\n\nИспользуйте этот режим для сравнения: даже в таком тексте детектор иногда увидит случайное совпадение с правилом метки.",
  hard: "На каждом шаге правило по ключу делит варианты продолжения на зелёные и красные. Вариант — это токен: слово, часть слова или знак препинания.\n\nМодель может выбрать только зелёный вариант. Метка получается заметной для детектора, но запрет подходящих красных вариантов может ухудшить текст.",
  soft: "На каждом шаге правило по ключу делит варианты продолжения на зелёные и красные. Зелёным повышают шанс, красные остаются разрешёнными.\n\nПараметр «Сила метки δ» задаёт величину преимущества. Детектор ищет избыток зелёных вариантов по сравнению с текстом без метки.",
  tournament: "Выбирает следующий токен — слово или часть слова — через несколько раундов отбора. Кандидаты берутся из вариантов модели.\n\nВ каждом раунде правило по ключу и предыдущим токенам даёт кандидату оценку 0 или 1. Оценка 1 побеждает 0; при равных оценках преимущества нет. Это служебные оценки, в текст цифры не вставляются.\n\nЧисло раундов задаёт параметр m. Детектор восстанавливает оценки написанных токенов и ищет повышенную долю единиц.",
};

const GREEN_DETECT_HINT = "Ищет избыток токенов из зелёного списка. Токен — слово, часть слова или знак препинания. Детектор восстанавливает список по ключу и предыдущим токенам, затем сравнивает долю зелёных с ожидаемой без метки.\n\nПодходит и для жёсткой, и для мягкой метки. Ключ, доля зелёных γ, длина контекста h и красный список должны совпадать с настройками генерации.";
const DETECT_HINT = {
  none: "Этот пункт не отключает проверку: здесь тоже работает детектор зелёного списка. Его можно использовать для опыта с текстом, созданным без метки.\n\nОн сравнивает долю зелёных токенов — слов или частей слов — с заданной долей γ. Случайный избыток возможен и без водяного знака.",
  hard: GREEN_DETECT_HINT,
  soft: GREEN_DETECT_HINT,
  tournament: "Проверяет оценки, которые правило турнира присваивает каждому токену — слову или части слова — в каждом раунде. Оценка 1 даёт преимущество при генерации, оценка 0 — нет.\n\nБез метки ожидается около половины единиц. Их избыток служит сигналом метки. Нужны тот же ключ, число раундов m, длина контекста h и красный список, что при генерации. Повтор одинакового токена после того же контекста учитывается один раз.",
};

function syncParamVisibility() {
  const mode = getMode();
  const visible = new Set(VISIBLE_PARAMS[mode]);
  document.querySelectorAll("[data-param]").forEach((el) => {
    el.hidden = !visible.has(el.dataset.param);
  });
  if (els.generateTip) els.generateTip.dataset.tip = MODE_HINT[mode] ?? "";
  document.querySelectorAll('input[name="wmMode"]').forEach((input) => {
    const label = input.closest("label");
    if (label) label.dataset.tip = MODE_HINT[input.value] ?? "";
  });
  syncDetectUi();
}

function syncDetectUi() {
  const mode = getDetectMode();
  const name = MODE_NAME[mode] ?? mode;
  const tip =
    `Ищет водяной знак в поле «Текст» выбранным способом: «${name}». Можно проверять созданный ответ, его отредактированную версию или вставленный текст.\n\n` +
    "Ключ берётся из поля «Ключ детектора». Длина контекста, число раундов турнира, доля зелёных и красный список — из настроек слева; используйте значения, с которыми текст помечали.\n\n" +
    "Для вставленного текста проверяются первые 2000 токенов — слов или частей слов. Начало используется как контекст. Результат относится к правилу этой программы: чужую метку с неизвестными настройками она не распознает.";
  if (els.detectTip) els.detectTip.dataset.tip = tip;
  document.querySelectorAll('input[name="detectMode"]').forEach((input) => {
    const label = input.closest("label");
    if (label) label.dataset.tip = DETECT_HINT[input.value] ?? "";
  });
}

const MODE_LABEL = { hard: "жёстким водяным знаком", soft: "мягким водяным знаком", tournament: "турниром SynthID" };

function clearOutput() {
  outputDirty = false;
  els.placeholder.hidden = true;
  els.promptEcho.textContent = "";
  els.tokens.textContent = "";
  setLegend(false);
  els.verdict.hidden = true;
  haveGeneration = false;
}

function resetOutput() {
  clearOutput();
  els.placeholder.hidden = false;
}

/* ── worker messages ── */

function handleMessage(msg) {
  if (DEBUG) console.debug("[worker]", msg);
  const a = active;
  switch (msg.type) {
    case "hello":
      a.hello = { app: msg.app, transformers: msg.transformers, webgpuApi: msg.webgpuApi };
      a.phase = "backend";
      log.push("hello", a.hello);
      break;

    case "phase":
      a.phase = msg.phase;
      break;

    case "backend":
      a.backend = msg;
      lastBackend = msg;
      log.push("backend", { device: msg.device, dtype: msg.dtype, note: msg.note });
      refreshModelOptions(msg);
      break;

    case "status":
      setStatus(msg.text);
      break;

    case "progress": {
      els.progress.hidden = false;
      const pct = Math.round(msg.progress ?? 0);
      const prev = a.loadPct ?? 0;
      // A new weight shard used to restart the bar at 0. Keep one rising scale.
      const shown = Math.max(prev, pct);
      a.loadPct = shown;
      els.progressFill.style.width = `${shown}%`;
      a.bytes = { loaded: msg.loaded, total: msg.total };
      const mb = (b) => (b / 1e6).toFixed(0);
      setStatus(shown >= 100 ? "Веса прочитаны, компилирую модель…" : `Загрузка модели: ${shown}% (${mb(msg.loaded)} / ${mb(msg.total)} МБ)`);
      break;
    }

    case "ready": {
      modelReady = true;
      modelLoading = false;
      els.activeModel.textContent = `Загружена: ${modelInfo(msg.modelId).name}`;
      a.phase = "ready";
      a.workerLog = msg.log;
      els.progress.hidden = true;
      els.model.value = msg.modelId; // programmatic .value does not fire "change"
      showBackendBadge(msg);
      const secs = Math.round((performance.now() - a.startedAt) / 100) / 10;
      log.push("ready", { modelId: msg.modelId, device: msg.device, dtype: msg.dtype, secs });
      if (a.retriedWasm) {
        setStatus("Модель готова, но на WASM: WebGPU не завёлся, будет медленнее.", "warn");
      } else {
        setStatus("Модель готова.");
      }
      if (DEBUG) buildReport(null).then(showDiagnostics, console.error);
      updateButtons();
      refreshCachedModels();
      break;
    }

    case "stream":
      if (streamSpan) {
        streamSpan.textContent += msg.text;
        els.output.scrollTop = els.output.scrollHeight;
      }
      break;

    case "token": {
      els.placeholder.hidden = true;
      if (streamSpan) {
        streamSpan.remove();
        streamSpan = null;
      }
      const span = document.createElement("span");
      span.className = "tok";
      span.textContent = msg.text;
      span.title = `Токен — фрагмент текста: слово, часть слова или знак препинания. Номер в словаре модели: ${msg.id}.`;
      els.tokens.appendChild(span);
      els.output.scrollTop = els.output.scrollHeight;
      break;
    }

    case "live":
      paintDetection(msg);
      break;

    case "generated": {
      generating = false;
      streamSpan = null;
      if (els.tokens.children.length !== msg.tokens.length) renderTokenChips(msg.tokens);
      els.promptEcho.textContent = "";
      haveGeneration = msg.tokens.length > 0;
      if (msg.detection) paintDetection(msg.detection);
      if (els.seedShown) els.seedShown.textContent = "42";
      setStatus(msg.interrupted ? "Генерация остановлена. Сид 42." : "Готово. Сид 42.");
      updateButtons();
      break;
    }

    case "detected": {
      if (msg.tokens) renderTokenChips(msg.tokens);
      paintDetection(msg);
      if (msg.note) els.verdictNote.textContent += msg.note;
      if (msg.pasted) {
        outputDirty = false;
        haveGeneration = (msg.tokens?.length ?? 0) > 0;

        setStatus("Проверен вставленный текст.");
        updateButtons();
      }
      break;
    }

    case "error":
      handleFailure({ phase: msg.phase, kind: msg.kind, raw: msg.raw, name: msg.name, stack: msg.stack, workerLog: msg.log });
      break;
  }
}

/** Uncaught error in the worker, or the worker script itself failed to load. */
function handleWorkerError(e) {
  const raw = e.message || "the worker script could not be loaded";
  const c = classifyError(raw, active?.backend?.device);
  const where = e.filename ? `${e.filename}:${e.lineno}:${e.colno}` : "";
  handleFailure({ phase: active?.phase ?? "starting", kind: c.kind, raw, name: "WorkerErrorEvent", stack: where, workerLog: active?.workerLog });
}

/* ── failure handling: WebGPU → WASM retry, otherwise explain + diagnostics ── */

async function handleFailure(f) {
  const a = active;
  if (!a) return;
  const failure = {
    ...f,
    modelId: a.modelId,
    cacheOnly: !!a.cacheOnly,
    requestedDevice: a.device ?? "auto",
    device: a.backend?.device ?? null,
    dtype: a.backend?.dtype ?? null,
    retriedWasm: !!a.retriedWasm,
    firstFailure: a.firstFailure ?? null,
    elapsedMs: Math.round(performance.now() - a.startedAt),
    bytes: a.bytes,
    hello: a.hello,
    backend: a.backend,
    workerLog: f.workerLog ?? a.workerLog,
  };
  log.push("failure", { phase: failure.phase, kind: failure.kind, raw: shorten(failure.raw, 200) });

  if (LOAD_PHASES.has(failure.phase) || !modelReady) {
    const name = modelInfo(a.modelId).name;
    const canRetryOnWasm =
      failure.device === "webgpu" &&
      !forcedDevice &&
      !a.retriedWasm &&
      !WEBGPU_ONLY.has(a.modelId) &&
      !["network", "unsupported"].includes(failure.kind);
    if (canRetryOnWasm) {
      log.push("retry-wasm");
      setStatus(`WebGPU сбой на этапе «${PHASE_TEXT[failure.phase] ?? failure.phase}» модели ${name}. Пробую WASM…`, "warn");
      showNote(`Ошибка WebGPU (${failure.kind}): ${shorten(failure.raw)}`);
      startLoad({
        modelId: a.modelId,
        device: "wasm",
        dtype: forcedDtype,
        retriedWasm: true,
        cacheOnly: a.cacheOnly,
        firstFailure: { phase: failure.phase, kind: failure.kind, raw: failure.raw, dtype: failure.dtype, elapsedMs: failure.elapsedMs, bytes: failure.bytes },
      });
      return;
    }

    retireWorker();
    modelLoading = false;
    els.activeModel.textContent = "Модель не загружена";
    els.progress.hidden = true;
    updateButtons();
    setStatus(`Не удалось загрузить ${name}. Собираю диагностику…`, "error");
    let report = null;
    try {
      report = await buildReport(failure);
    } catch (err) {
      console.error("[watermarking-playground] diagnostics failed", err);
    }
    setStatus(describeFailure(failure, report), "error");
    showNote(adviceFor(failure, report));
    if (report) showDiagnostics(report);
    refreshCachedModels();
    console.error("[watermarking-playground] load failed", report ?? failure);
    return;
  }

  if (failure.phase === "generate") {
    generating = false;
    streamSpan = null;
    if (failure.kind === "oom" && crashReloads < 1) {
      // The ONNX Runtime instance is dead after an abort: reload the same model in a fresh worker.
      crashReloads++;
      const name = modelInfo(a.modelId).name;
      log.push("reload-after-crash");
      setStatus(`Среда упала во время генерации (${shorten(failure.raw)}). Перезагружаю ${name}. Попробуйте меньше токенов.`, "warn");
      startLoad({ modelId: a.modelId, device: a.backend?.device ?? a.device, dtype: forcedDtype, keepOutput: true, cacheOnly: a.cacheOnly });
      return;
    }
    setStatus(`Генерация не удалась: ${failure.raw}`, "error");
    updateButtons();
    return;
  }

  setStatus(`${failure.phase === "detect" ? "Проверка" : "Операция"} не удалась: ${failure.raw}`, "error");
}

function describeFailure(f, report) {
  const name = modelInfo(f.modelId).name;
  const where = PHASE_TEXT[f.phase] ?? f.phase;
  const file = sizeMB(f.modelId, f.dtype);
  const need = estimatedLoadMB(f.modelId, f.dtype);
  const heap = report?.heapProbeMB;
  const on = f.device ? ` на ${f.device === "webgpu" ? "WebGPU" : "WASM"} (${f.dtype}${file ? `, файл ${formatMB(file)}` : ""})` : "";
  switch (f.kind) {
    case "oom":
      if (f.modelId === "onnx-community/Qwen3-1.7B-ONNX") {
        return (
          `${name} не удалось собрать на этапе «${where}»${on}. ` +
          "Здесь важна память графа ONNX в браузерном рантайме: 32 ГБ системной RAM не увеличивают отдельный лимит WebAssembly. " +
          "Gemma и Qwen используют разные ONNX-графы и рабочие пути, поэтому успешный запуск Gemma не гарантирует, что Qwen поместится. " +
          "Для Qwen3-1.7B отключён откат на более тяжёлый Q8 WASM; попробуйте Qwen3-0.6B."
        );
      }
      return (
        `${name} не влезла в память браузера на этапе «${where}»${on}. ` +
        `На загрузку нужно ≈${formatMB(need)} памяти WebAssembly (граф собирается, пока веса ещё лежат в памяти)` +
        (heap
          ? `, а этот браузер смог вырастить кучу только до ≈${formatMB(heap)}.`
          : ". У WebAssembly потолок 4 ГБ, на телефонах ещё меньше.") +
        " Эта модель и так самая большая, которую сюда можно поставить."
      );
    case "network":
      if (f.cacheOnly) return `В кэше не хватает файлов ${name}. Нажмите «Скачать и загрузить»: недостающие файлы скачаются, затем модель загрузится в память.`;
      return `Не удалось загрузить ${name} на этапе «${where}»: ${shorten(f.raw)}. Это офлайн-копия: веса должны быть в models/, рантайм в vendor4/.`;
    case "unsupported":
      return f.raw;
    case "webgpu":
      return (
        `WebGPU отказала на этапе «${where}» модели ${name}: ${shorten(f.raw)}.` +
        (forcedDevice === "webgpu" ? " Уберите ?device=webgpu из адреса, чтобы можно было откатиться на WASM." : "")
      );
    default:
      if (f.phase === "starting") {
        return (
          `Среда не запустилась${f.raw ? ` (${shorten(f.raw)})` : ""}. ` +
          "Открывайте страницу через start.bat, не двойным щелчком по файлу. Нужны WebAssembly SIMD и ES-модули в воркере, а в vendor4/ должны лежать файлы ONNX Runtime."
        );
      }
      return `Не удалось загрузить ${name} на этапе «${where}»${on}: ${shorten(f.raw)}`;
  }
}

function adviceFor(f, report) {
  const parts = [];
  if (f.firstFailure) {
    parts.push(`Сначала пробовали WebGPU, там тоже ошибка (${f.firstFailure.kind}: ${shorten(f.firstFailure.raw, 120)}).`);
  }
  if (f.kind === "oom") {
    const dtype = f.dtype ?? "q8";
    const heap = report?.heapProbeMB;
    const candidates = MODELS.filter((m) => !m.webgpuOnly && m.id !== f.modelId)
      .map((m) => ({ name: m.name, need: estimatedLoadMB(m.id, dtype) }))
      .filter((c) => c.need != null && (!heap || c.need <= heap));
    if (candidates.length) {
      parts.push(`Сюда могли бы влезть: ${candidates.map((c) => `${c.name} (≈${formatMB(c.need)})`).join(", ")}.`);
    } else {
      parts.push("Даже маленькая модель может не влезть. Нужен настольный Chrome или Edge с WebGPU.");
    }
  }
  return parts.join(" ");
}

/* ── diagnostics ── */

async function buildReport(failure) {
  const environment = await collectEnvironment();
  const heapProbeMB = failure?.kind === "oom" || DEBUG ? probeWasmHeapMB() : null;
  const a = active;
  const modelId = failure?.modelId ?? a?.modelId ?? null;
  const dtype = failure?.dtype ?? a?.backend?.dtype ?? null;
  return {
    app: APP_VERSION,
    time: new Date().toISOString(),
    url: location.href,
    result: failure ? "failure" : "success",
    failure: failure
      ? { phase: failure.phase, kind: failure.kind, raw: failure.raw, name: failure.name, stack: failure.stack, elapsedMs: failure.elapsedMs, bytes: failure.bytes, firstFailure: failure.firstFailure }
      : null,
    attempt: {
      modelId,
      requestedDevice: failure?.requestedDevice ?? a?.device ?? "auto",
      device: failure?.device ?? a?.backend?.device ?? null,
      dtype,
      fileMB: sizeMB(modelId, dtype),
      estimatedLoadMB: estimatedLoadMB(modelId, dtype),
      retriedWasm: failure?.retriedWasm ?? !!a?.retriedWasm,
      elapsedMs: failure?.elapsedMs ?? (a ? Math.round(performance.now() - a.startedAt) : null),
    },
    runtime: failure?.hello ?? a?.hello ?? null,
    backend: failure?.backend ?? a?.backend ?? lastBackend,
    heapProbeMB,
    environment,
    log: log.entries(),
    workerLog: failure?.workerLog ?? a?.workerLog ?? null,
  };
}

function showDiagnostics(report) {
  const f = report.failure;
  const at = report.attempt;
  const head = [
    `Песочница водяных знаков — диагностика (сборка ${report.app}, ${report.time})`,
    f ? `FAILURE while ${f.phase} · kind=${f.kind} · ${shorten(f.raw, 300)}` : "SUCCESS",
    `model=${at.modelId} device=${at.device ?? "?"} (requested ${at.requestedDevice}) dtype=${at.dtype ?? "?"} file≈${formatMB(at.fileMB)} needs≈${formatMB(at.estimatedLoadMB)} elapsed=${at.elapsedMs} ms`,
    report.heapProbeMB != null ? `WebAssembly heap could grow to ≈${formatMB(report.heapProbeMB)}` : null,
    "",
  ]
    .filter((l) => l !== null)
    .join("\n");
  els.diagText.textContent = `${head}${JSON.stringify(report, null, 2)}`;
  els.diag.hidden = false;
  els.diag.open = !!f;
}

function hideDiagnostics() {
  els.diag.hidden = true;
  els.diag.open = false;
  els.diagText.textContent = "";
}

els.copyDiag.addEventListener("click", async () => {
  const label = els.copyDiag.textContent;
  try {
    await navigator.clipboard.writeText(els.diagText.textContent);
    els.copyDiag.textContent = "Скопировано";
  } catch {
    const range = document.createRange();
    range.selectNodeContents(els.diagText);
    const sel = getSelection();
    sel.removeAllRanges();
    sel.addRange(range);
    els.copyDiag.textContent = "Выделено — нажмите Ctrl+C";
  }
  setTimeout(() => {
    els.copyDiag.textContent = label;
  }, 2000);
});

/* ── verdict rendering ── */

function renderVerdict({ scheme, z, pValue, greenCount, T, totalT = T, gamma, meanG, m, h, expectedG }) {
  els.verdict.hidden = false;
  const tournament = scheme === "tournament";
  const reference = tournament ? (Number(expectedG) || expectedTournamentMean(m)) : 1;
  const budget = Math.max(1e-4, reference - 0.5);
  const g = Number.isFinite(meanG) ? meanG : 0.5;
  const checks = (T || 0) * (m || 1);
  const need = Math.ceil(1 / (budget * budget));
  const ready = tournament ? checks >= need : T * gamma >= 5 && T * (1 - gamma) >= 5;
  const raw = tournament
    ? Math.min(1, Math.max(0, (g - 0.5) / budget)) * Math.min(1, checks / need)
    : ((T ? greenCount / T : gamma) - gamma) / Math.max(1e-9, 1 - gamma);
  const pct = Math.max(0, Math.min(100, raw * 100));
  const pctText = pct >= 99.5 ? "100%" : pct < 10 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;
  const labels = !ready
    ? ["Мало данных для вывода", "mid"]
    : pct >= 70 ? ["Сильный сигнал метки", "pos"]
    : pct >= 30 ? ["Слабый сигнал метки", "mid"]
    : ["Нет выраженного сигнала", "neg"];
  const repeatNote = totalT > T ? ` Повторных сочетаний исключено: ${totalT - T}.` : "";
  const note = (tournament
    ? `Доля единиц: ${(g * 100).toFixed(1)}%; без метки ожидается около 50%. Учтено токенов: ${T}, раундов на токен: ${m}.`
    : `Зелёных токенов: ${T ? (greenCount / T * 100).toFixed(1) : "0"}%; без метки ожидается около ${(gamma * 100).toFixed(1)}%. Учтено токенов: ${T}.`) + repeatNote;

  els.verdictLabel.textContent = labels[0];
  els.verdictLabel.className = `verdict-label ${labels[1]}`;
  els.verdictConf.hidden = false;
  els.meterFill.parentElement.hidden = false;
  els.verdictConf.textContent = `сила ${pctText}`;
  els.meterFill.style.width = `${pct.toFixed(1)}%`;
  const scaleTip = tournament
    ? `Показывает, насколько доля оценок 1 поднялась над уровнем 50%, ожидаемым без метки. Оценку 1 правило турнира даёт вариантам, которым благоприятствует при выборе продолжения.\n\n100% на шкале соответствует ориентиру ${(reference * 100).toFixed(1)}% единиц при ${m} раундах. Он рассчитан на условном наборе вариантов, а не на вашей модели; это не предел. Для короткого текста процент дополнительно снижается.`
    : `Показывает, насколько доля зелёных токенов поднялась над ожидаемыми без метки ${(gamma * 100).toFixed(1)}%. Зелёные — варианты, которым правило метки даёт преимущество.\n\nШкала идёт от 0% при ожидаемой доле до 100%, когда все учтённые токены зелёные. Например, при ожидаемых 50% и наблюдаемых 75% сила равна 50%.`;
  els.verdictConf.dataset.tip = scaleTip +
    "\n\nЭто условная сила сигнала, не вероятность авторства ИИ. При достаточном объёме данных: от 70% — сильный сигнал, от 30% — слабый. Статистическая оценка z показана отдельно и не определяет эту подпись.";

  els.statScheme.textContent = tournament ? `турнир · ${m} раундов` : "зелёный список";
  els.statFirstLabel.textContent = tournament ? "доля единиц" : "зелёные токены";
  els.statGreen.textContent = tournament
    ? `${(g * 100).toFixed(1)}% (${T} × ${m})`
    : `${greenCount} / ${T} (${T ? (greenCount / T * 100).toFixed(0) : "0"}%)`;
  els.statZ.textContent = T ? z.toFixed(2) : "—";
  els.statP.textContent = T ? fmtP(pValue) : "—";
  els.verdictNote.textContent = note;
  setVerdictTips({ tournament, z, pValue, greenCount, T, totalT, gamma, meanG: g, m, h, ready });
}

function setVerdictTips({ tournament, greenCount, T, totalT, gamma, meanG, m, h, ready }) {
  const tip = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.dataset.tip = text;
  };
  const counting = `Учтено ${T} из ${totalT} проверяемых токенов. Одинаковый токен после тех же предыдущих токенов учитывается один раз: повтор не даёт нового свидетельства метки.`;
  const baseline = tournament
    ? "доля оценок 1 — около 50%"
    : `доля зелёных токенов — около ${(gamma * 100).toFixed(1)}%`;
  if (tournament) {
    tip("tip-scheme",
      `Выбран детектор турнирной метки. Он восстанавливает служебные оценки 0 и 1 по ключу, затем ищет избыток единиц.\n\nРаундов на токен: ${m} (параметр m). Предыдущих токенов в расчёте: ${h} (параметр h). Токен — слово или часть слова. Эти параметры и ключ должны совпадать с генерацией.`
    );
    tip("tip-green",
      `В каждом раунде правило турнира присваивает токену оценку 0 или 1. При генерации 1 даёт преимущество перед 0. Это результат вычисления по ключу и контексту, а не цифры, спрятанные в тексте.\n\nСейчас единицы составляют ${(meanG * 100).toFixed(1)}% из ${T * m} оценок: ${T} токенов × ${m} раундов. Без метки ожидается около 50%.\n\n` + counting
    );
  } else {
    tip("tip-scheme",
      `Выбран детектор зелёного списка. Он ищет варианты продолжения, которым правило метки даёт преимущество. Проверка одинакова для жёсткого и мягкого режимов.\n\nОжидаемая доля зелёных без метки: ${(gamma * 100).toFixed(1)}% (параметр γ). Предыдущих токенов в расчёте: ${h} (параметр h). Токен — слово или часть слова. Эти параметры и ключ должны совпадать с генерацией.`
    );
    tip("tip-green",
      `Зелёный токен — фрагмент текста, которому правило метки давало преимущество в этой позиции. Детектор восстанавливает цвет по ключу и предыдущим токенам.\n\nЗелёных: ${greenCount} из ${T}. Без метки ожидается около ${(gamma * 100).toFixed(1)}%. Превышение этого уровня служит сигналом метки.\n\n` + counting
    );
  }
  tip("tip-z",
    `z-оценка сравнивает найденный сигнал с обычным случайным разбросом. Без метки ожидается: ${baseline}. Чем больше положительное z, тем необычнее избыток для текста без метки.\n\nz около 0 — близко к ожидаемому; отрицательное z — ниже него; z = 4 — превышение на четыре стандартных отклонения, то есть четыре меры разброса.\n\nЭто приближённая статистика. На коротких текстах она ненадёжна. Подпись над шкалой выбирается по силе сигнала, а не по порогу z.`
  );
  tip("tip-p",
    `p-значение оценивает, как часто без водяного знака случайно получился бы такой же или более сильный сигнал. Оно вычисляется из z-оценки, которая измеряет превышение над ожидаемым уровнем.\n\nНапример, p = 0.01 означает примерно 1 такой случай на 100 при допущениях расчёта. Это не вероятность того, что текст написал человек или ИИ. Запись 1e-15 означает единицу, делённую на 10 в пятнадцатой степени.\n\nЧем меньше p, тем необычнее результат без метки. Оценка приближённая; на коротких текстах ей нельзя доверять как точной вероятности.`
  );
  tip("verdict-note",
    (tournament
      ? `В турнире каждый токен — слово или часть слова — получает оценку 0 или 1 в каждом раунде. Оценка 1 даёт преимущество при выборе продолжения; сами цифры в текст не вставляются.\n\nЗдесь единиц ${(meanG * 100).toFixed(1)}%. Без метки ожидается около 50%; избыток служит её сигналом. Раундов на токен: ${m}. В настройках это число обозначено m.`
      : `Зелёные токены — слова или части слов, которым правило метки даёт преимущество. Здесь их ${T ? (greenCount / T * 100).toFixed(1) : "0"}%. Без метки ожидается около ${(gamma * 100).toFixed(1)}%: эту долю задаёт параметр γ.`) +
    `\n\n${counting}\n\n` +
    (!ready ? "Пока данных мало: результат может сильно измениться при добавлении текста. " : "") +
    "Процент над шкалой показывает условную силу сигнала. Отсутствие сигнала не доказывает, что текст написал человек: метку могли не ставить, изменить текст или проверить с другим ключом."
  );
}

function fmtP(p) {
  if (p < 1e-15) return "< 1e-15";
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

/* ── user actions ── */

els.generateBtn.addEventListener("click", () => {
  if (!modelReady || !active?.worker) {
    setStatus("Сначала загрузите модель в память.", "error");
    return;
  }
  const p = readParams();
  if (!p.prompt.trim()) {
    setStatus("Сначала напишите промпт.", "error");
    return;
  }
  clearOutput();
  generating = true;
  updateButtons();
  setStatus(p.mode === "none" ? "Генерация…" : `Генерация с ${MODE_LABEL[p.mode]}…`);

  streamSpan = document.createElement("span");
  els.tokens.appendChild(streamSpan);
  els.promptEcho.textContent = "";

  send({ type: "generate", params: { ...p, detect: detectParams() } });
});

els.stopBtn.addEventListener("click", () => send({ type: "interrupt" }));

els.loadCachedModel.addEventListener("click", () => {
  crashReloads = 0;
  startLoad({ modelId: els.model.value, device: forcedDevice, dtype: forcedDtype, cacheOnly: true });
});

els.downloadModel.addEventListener("click", downloadSelectedModel);

function detectParams() {
  const mode = getDetectMode();
  const { gamma, h, m, redWords } = readParams();
  return { scheme: mode === "tournament" ? "tournament" : "greenlist", gamma, h, m, redWords, key: els.detectKey.value };
}

function setLegend(on) {
  els.legend.classList.toggle("is-off", !on);
}

function paintDetection(msg) {
  lastPaint = msg;
  const chips = els.tokens.children;
  const tournament = msg.scheme === "tournament";
  const color = els.colorText?.checked !== false;
  const colorFrom = msg.colorFrom ?? 0;
  for (let i = 0; i < chips.length; i++) {
    const flag = i - colorFrom;
    const known = flag >= 0 && flag < (msg.flags?.length ?? 0);
    chips[i].classList.toggle("green", color && known && !!msg.flags[flag]);
    chips[i].classList.toggle("red", color && known && !msg.flags[flag]);
    const base = chips[i].title.split(" · ")[0];
    chips[i].title = !color || !known ? base : tournament
      ? `${base} · В ${(msg.perTokenScore[flag] * 100).toFixed(0)}% раундов правило турнира дало этому токену оценку 1 — преимущество при выборе. Зелёный цвет означает не меньше половины таких раундов, красный — меньше половины.`
      : `${base} · ${msg.flags[flag] ? "Зелёный: правило метки даёт этому варианту преимущество." : "Красный: правило метки не даёт этому варианту преимущества."} Цвет одного токена не определяет результат проверки всего текста.`;
  }
  els.legendHi.textContent = tournament ? "единиц ≥ 50%" : "зелёный список";
  els.legendLo.textContent = tournament ? "единиц < 50%" : "красный список";
  els.colorText.closest("label").dataset.tip = tournament
    ? "Показывает оценки отдельных токенов — слов или частей слов. В каждом раунде правило турнира даёт токену 0 или 1; единица даёт преимущество при генерации. Зелёный цвет: единиц не меньше половины, красный: меньше половины. Цвет не означает, что слово правильное или ошибочное. Выключение раскраски не меняет проверку."
    : "Показывает, какие токены — слова или части слов — входят в зелёный и красный списки по ключу детектора. Зелёные получают преимущество при генерации с меткой. Отдельное зелёное слово встречается и без метки; важна общая доля. Выключение раскраски не меняет проверку.";
  setLegend(color);
  renderVerdict(msg);
}

function renderTokenChips(tokens) {
  for (const node of [...els.output.childNodes]) {
    if (node !== els.placeholder && node !== els.promptEcho && node !== els.tokens) node.remove();
  }
  els.placeholder.hidden = true;
  els.promptEcho.textContent = "";
  els.tokens.textContent = "";
  setLegend(false);
  for (const tok of tokens) {
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = tok.text;
    span.title = `Токен — фрагмент текста: слово, часть слова или знак препинания. Номер в словаре модели: ${tok.id}.`;
    els.tokens.appendChild(span);
  }
}

els.colorText.addEventListener("change", () => {
  if (lastPaint) paintDetection(lastPaint);
  else els.legend.hidden = true;
});

els.output.addEventListener("pointerdown", () => {
  if (!els.placeholder.hidden) els.placeholder.hidden = true;
});
els.output.addEventListener("input", () => {
  outputDirty = true;
  els.placeholder.hidden = true;
  updateButtons();
});
els.output.addEventListener("paste", (e) => {
  e.preventDefault();
  const text = e.clipboardData?.getData("text/plain") ?? "";
  els.placeholder.hidden = true;
  const sel = getSelection();
  if (!sel?.rangeCount) {
    els.tokens.appendChild(document.createTextNode(text));
  } else {
    sel.deleteFromDocument();
    sel.getRangeAt(0).insertNode(document.createTextNode(text));
    sel.collapseToEnd();
  }
  outputDirty = true;
  updateButtons();
});

function textToCheck() {
  if (!outputDirty) return "";
  return els.output.innerText.replace(/\u00a0/g, " ").trim();
}

els.detectBtn.addEventListener("click", () => {
  const text = textToCheck();
  if (text) {
    setStatus("Проверяю вставленный текст…");
    send({ type: "detectText", text, params: detectParams() });
    return;
  }
  if (!haveGeneration) {
    setStatus("Сначала сгенерируйте ответ или вставьте текст.", "error");
    return;
  }
  setStatus("Проверяю последний ответ…");
  send({ type: "detect", params: detectParams() });
});

els.model.addEventListener("change", () => {
  const modelId = els.model.value;
  crashReloads = 0;
  log.push("user-select", { modelId });
  retireWorker();
  modelReady = false;
  modelLoading = false;
  els.activeModel.textContent = "Модель не загружена";
  generating = false;
  haveGeneration = false;
  resetOutput();
  els.backendBadge.hidden = true;
  els.progress.hidden = true;
  updateButtons();
  showNote("");
  hideDiagnostics();
  setStatus(`Выбрана ${modelInfo(modelId).name}. Скачайте её или загрузите из кэша.`);
  if (WEBGPU_ONLY.has(modelId) && forcedDevice !== "webgpu" && lastBackend && lastBackend.device !== "webgpu") {
    setStatus(`${modelInfo(modelId).name} нужна WebGPU, а в этом браузере её нет (сейчас WASM). Возьмите Chrome или Edge.`, "error");
  }
});

document.querySelectorAll('input[name="wmMode"]').forEach((r) => r.addEventListener("change", () => {
  const detect = document.querySelector(`input[name="detectMode"][value="${r.value}"]`);
  if (detect) detect.checked = true;
  syncParamVisibility();
}));
document.querySelectorAll('input[name="detectMode"]').forEach((r) => r.addEventListener("change", syncDetectUi));

els.prompt.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter" && !els.generateBtn.disabled) {
    els.generateBtn.click();
  }
});

/* ── init ── */
const hintPop = document.getElementById("hint-pop");
let hintAnchor = null;
let hintHideTimer;
function tipHost(node) {
  const el = node?.closest?.("[data-tip], label");
  if (!el) return null;
  if (el.dataset.tip) return el;
  const inner = el.querySelector("[data-tip]");
  return inner?.dataset.tip ? inner : null;
}
function placeHint(anchor) {
  const text = anchor.dataset.tip;
  if (!text) return;
  clearTimeout(hintHideTimer);
  hintAnchor = anchor;
  hintPop.hidden = false;
  hintPop.textContent = text;
  const margin = 8;
  const r = anchor.getBoundingClientRect();
  const w = hintPop.offsetWidth;
  const h = hintPop.offsetHeight;
  let left = Math.min(r.left, window.innerWidth - margin - w);
  left = Math.max(margin, left);
  let top = r.top - h - margin;
  if (top < margin) top = Math.min(r.bottom + margin, window.innerHeight - margin - h);
  top = Math.max(margin, top);
  hintPop.style.left = `${left}px`;
  hintPop.style.top = `${top}px`;
}
function hideHint() {
  clearTimeout(hintHideTimer);
  hintPop.hidden = true;
  hintAnchor = null;
}
document.addEventListener("mouseover", (e) => {
  if (hintPop.contains(e.target)) {
    clearTimeout(hintHideTimer);
    return;
  }
  const host = tipHost(e.target);
  if (!host) return;
  placeHint(host);
});
document.addEventListener("mouseout", (e) => {
  const host = tipHost(e.target);
  if (!host && !hintPop.contains(e.target)) return;
  if (hintAnchor?.contains(e.relatedTarget) || hintPop.contains(e.relatedTarget)) return;
  // Allow crossing the gap into a long tooltip to scroll its text.
  hintHideTimer = setTimeout(hideHint, 180);
});
document.addEventListener("focusin", (e) => {
  const host = e.target.closest?.("[data-tip]");
  if (host?.dataset.tip) placeHint(host);
});
document.addEventListener("focusout", hideHint);
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideHint();
});
window.addEventListener("scroll", (e) => {
  if (e.target === hintPop) return;
  if (hintAnchor?.contains(document.activeElement)) placeHint(hintAnchor);
  else hideHint();
}, true);
window.addEventListener("resize", hideHint);

syncParamVisibility();
refreshModelOptions(lastBackend);
updateButtons();
setStatus("Выберите модель: нажмите «Скачать и загрузить» или загрузите уже скачанную из кэша.");
refreshCachedModels();
