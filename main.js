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
} from "./models.js";
import { expectedTournamentMean, layerWeights } from "./watermark.js";
import { displayParts } from "./output-text.js";

const $ = (id) => document.getElementById(id);

const els = {
  model: $("model"),
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
  statMean: $("stat-mean"),
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
  verdictNoteHelp: $("verdict-note-help"),
  strengthChart: $("strength-chart"),
  strengthChartPlot: $("strength-chart-plot"),
  strengthChartHelp: $("strength-chart-help"),
  wmModeHelp: $("wm-mode-help"),
  detectModeHelp: $("detect-mode-help"),
  colorHelp: $("color-help"),
  verdictConfHelp: $("verdict-conf-help"),
  copyOutput: $("copy-output"),
  copyVerdict: $("copy-verdict"),
};

let modelReady = false;
let modelLoading = false;
let downloadingModel = false;
let generating = false;
let haveGeneration = false;
let outputDirty = false;
let outputEditVersion = 0;
let detectionRequestId = 0;
let pendingTextDetection = null;
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
    ? "worker-v4.js"
    : runtime === "v4next"
      ? "worker-next.js"
      : "worker.js";
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
    topK: Math.round(clamp(els.topK.value, 0, 1000, 50)),
    topP: clamp(els.topP.value, 0.05, 1, 1),
    seed: 42,
    gamma: clamp(els.gamma.value, 0.05, 0.95, 0.5),
    delta: clamp(els.delta.value, 0, 15, 2),
    h: Math.round(clamp(els.ctxWidth.value, 1, 8, 4)),
    m: Math.round(clamp(els.layers.value, 1, 30, 30)),
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
  els.downloadModel.disabled = generating || modelLoading || downloadingModel;
  for (const button of els.cachedModelList.querySelectorAll("button")) {
    button.disabled = generating || modelLoading || downloadingModel;
  }
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
  for (const option of [...els.model.options]) {
    if (option.dataset.cachedOnly && !cachedModels.has(option.value)) option.remove();
  }
  els.cachedModelList.replaceChildren();
  if (!cachedModels.size) {
    els.cachedModelList.textContent = "Скачанных моделей пока нет. Выберите модель выше и нажмите «Скачать выбранную модель и запустить».";
    return;
  }
  for (const entry of MODELS) {
    const cached = cachedModels.get(entry.id);
    if (!cached) continue;
    if (![...els.model.options].some((option) => option.value === entry.id)) {
      const option = new Option(entry.name, entry.id);
      option.dataset.cachedOnly = "true";
      els.model.add(option);
    }
    const row = document.createElement("div");
    row.className = "cached-model-row";
    const label = document.createElement("span");
    const variant = cached.variants.length ? cached.variants.join(", ").toUpperCase() : "ONNX";
    const size = cached.size ? ` · ~${(cached.size / 1000).toFixed(1).replace(".", ",")} ГБ` : "";
    label.textContent = `${entry.name} (${variant}${size})`;
    const actions = document.createElement("div");
    actions.className = "cached-model-actions";
    const load = document.createElement("button");
    load.type = "button";
    load.className = "btn primary small";
    load.textContent = "Загрузить";
    load.setAttribute("aria-label", `Загрузить ${entry.name} из кэша`);
    load.addEventListener("click", () => {
      if (generating || modelLoading || downloadingModel) return;
      if (els.model.value !== entry.id) {
        els.model.value = entry.id;
        els.model.dispatchEvent(new Event("change"));
      }
      crashReloads = 0;
      startLoad({ modelId: entry.id, device: forcedDevice, dtype: forcedDtype, cacheOnly: true });
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "btn secondary small";
    remove.textContent = "Удалить из кэша";
    remove.setAttribute("aria-label", `Удалить ${entry.name} из кэша`);
    remove.addEventListener("click", () => deleteCachedModel(entry));
    actions.append(load, remove);
    row.append(label, actions);
    els.cachedModelList.append(row);
  }
  updateButtons();
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
  if (els.wmModeHelp) els.wmModeHelp.dataset.tip = MODE_HINT[mode] ?? "";
  syncDetectUi();
}

function syncDetectUi() {
  const mode = getDetectMode();
  const name = MODE_NAME[mode] ?? mode;
  const tip =
    `Ищет водяной знак в поле «Текст» выбранным способом: «${name}». Можно проверять созданный ответ, его отредактированную версию или вставленный текст.\n\n` +
    "Ключ берётся из поля «Ключ детектора». Длина контекста, число раундов турнира, доля зелёных и красный список — из настроек слева; используйте значения, с которыми текст помечали.\n\n" +
    "Для вставленного текста проверяются и подсвечиваются все токены (не более 200 000). «Лимит ответа» ограничивает только генерацию. Начало используется как контекст. Результат относится к правилу этой программы: чужую метку с неизвестными настройками она не распознает.";
  if (els.detectTip) els.detectTip.dataset.tip = tip;
  if (els.detectModeHelp) els.detectModeHelp.dataset.tip = DETECT_HINT[mode] ?? "";
}

const MODE_LABEL = { hard: "жёстким водяным знаком", soft: "мягким водяным знаком", tournament: "турниром SynthID" };

function clearOutput() {
  outputDirty = false;
  outputEditVersion++;
  pendingTextDetection = null;
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
      if (msg.pasted) {
        if (!pendingTextDetection || msg.requestId !== pendingTextDetection.id || outputEditVersion !== pendingTextDetection.editVersion) break;
        renderTokenChips(msg.tokens, pendingTextDetection.text);
        pendingTextDetection = null;
      } else if (msg.tokens) {
        renderTokenChips(msg.tokens);
      }
      paintDetection(msg);
      if (msg.note) {
        els.verdictNote.textContent += msg.note;
        els.verdictNote.hidden = false;
      }
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
      if (f.cacheOnly) return `В кэше не хватает файлов ${name}. Выберите её выше и нажмите «Скачать выбранную модель и запустить»: недостающие файлы скачаются, затем модель загрузится в память.`;
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

/* ── copy buttons: the text and the detector readings ── */

/** Copy to the clipboard and flash the button's check icon. */
async function copyWithFeedback(button, text) {
  let ok = false;
  try {
    await navigator.clipboard.writeText(text);
    ok = true;
  } catch {
    // Clipboard API unavailable (non-secure context): hidden textarea fallback.
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    ta.remove();
  }
  if (ok) {
    button.classList.add("copied");
    setTimeout(() => button.classList.remove("copied"), 1600);
  }
}

/** Plain text of the answer as shown in the output card, without colours. */
function outputPlainText() {
  return outputTextFromDom().trim();
}

/**
 * Compact plain-text summary of the detector readings, built from the very
 * numbers on screen so the paste always matches the display:
 *
 *   Вердикт: Сильный сигнал метки · сила 100%
 *   детектор: SynthID · 15 раундов
 *   взвешенное среднее: 54.2% (399 × 15)
 *   среднее: 53.9%
 *   z-оценка: 6.01
 *   p-значение: 9.3e-10
 */
function verdictPlainText() {
  if (!lastPaint) return "";
  const lines = [
    `Вердикт: ${els.verdictLabel.textContent} · ${els.verdictConf.textContent}`,
    `детектор: ${els.statScheme.textContent}`,
    `${els.statFirstLabel.textContent}: ${els.statGreen.textContent}`,
  ];
  if (lastPaint.scheme === "tournament") lines.push(`среднее: ${els.statMean.textContent}`);
  lines.push(`z-оценка: ${els.statZ.textContent}`, `p-значение: ${els.statP.textContent}`);
  return lines.join("\n");
}

els.copyOutput.addEventListener("click", () => {
  const text = outputPlainText();
  if (text) copyWithFeedback(els.copyOutput, text);
});

els.copyVerdict.addEventListener("click", () => {
  const text = verdictPlainText();
  if (text) copyWithFeedback(els.copyVerdict, text);
});

/* ── verdict rendering ── */

function renderVerdict({ scheme, z, pValue, greenCount, T, totalT = T, gamma, meanG, weightedMeanG, m, h, expectedG, expectedSource }) {
  els.verdict.hidden = false;
  if (els.copyVerdict) els.copyVerdict.disabled = false;
  const tournament = scheme === "tournament";
  const plainMean = Number.isFinite(meanG) ? meanG : 0.5;
  // Headline number of the tournament scheme: the official Weighted Mean score.
  const g = tournament ? (Number.isFinite(weightedMeanG) ? weightedMeanG : plainMean) : plainMean;
  const zVal = Number.isFinite(z) ? z : 0;
  const gammaVal = Number.isFinite(gamma) ? gamma : 0.5;

  // Справочный ориентир (только для подсказок, НЕ критерий вердикта):
  // официальная формула при известном словаре, иначе синтетическое распределение.
  const reference = tournament ? Number(expectedG) || expectedTournamentMean(m) : 1;

  /* Статистическое свидетельство.
   *
   * z — основная величина, по которой строится и вердикт, и шкала. Нулевая
   * модель трактует каждое учтённое наблюдение как независимый бит хэша:
   * раунды хэшируются разными константами, а повторяющиеся контексты
   * исключены маской, поэтому турнир даёт trials = T·m испытаний Бернулли(1/2),
   * зелёный список — T испытаний Бернулли(γ). Предпосылки номинальные:
   * точная частота ложных срабатываний требует эмпирической калибровки
   * на корпусе немаркированных текстов.
   *
   * Достаточность данных: не менее 25 неповторяющихся токенов и достаточно
   * ожидаемых успехов/неудач для нормальной аппроксимации (≥ 20 бит у
   * «честной монеты» турнира; ≥ 5 зелёных и красных у списка). Недостаток
   * данных пропорционально гасит шкалу, чтобы один токен с высоким
   * расчётным z не выглядел убедительным. */
  const trials = tournament ? (T || 0) * (m || 1) : T || 0;
  const sufficiency = Math.max(0, Math.min(1, tournament
    ? Math.min((T || 0) / 25, trials / 20)
    : Math.min((T || 0) / 25, (T * gammaVal) / 5, (T * (1 - gammaVal)) / 5)));
  const ready = sufficiency >= 1;

  // Вердикт — по номинальным односторонним порогам z:
  // z ≥ 2 (p ≈ 0.023) — слабый сигнал, z ≥ 4 (p ≈ 3.2e-5) — сильный.
  const labels = !ready
    ? ["Мало данных для вывода", "mid"]
    : zVal >= 4 ? ["Сильный сигнал метки", "pos"]
    : zVal >= 2 ? ["Слабый сигнал метки", "mid"]
    : ["Нет выраженного сигнала", "neg"];

  /* Шкала «сила» — условная переформулировка z, а НЕ вероятность:
   * линейна по z (0% при z = 0, ≈33% при z = 2, ≈67% при z = 4, 100% при z = 6,
   * что соответствует номинальному p ≈ 1e-9) и пропорционально уменьшена
   * при нехватке данных. Отношение среднего g к эталону не используется. */
  const pct = Math.max(0, Math.min(100, (100 * zVal) / 6)) * sufficiency;
  const pctText = pct >= 99.5 ? "100%" : pct < 10 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;

  const repeatNote = totalT > T
    ? (tournament
      ? `Исключено повторных контекстов: ${totalT - T}.`
      : `Исключено повторных сочетаний: ${totalT - T}.`)
    : "";
  // The note stays minimal: every number and caveat already lives in the
  // stats table and its tooltips. Only operational facts are shown here.
  const note = !ready
    ? "Данных мало: результат может сильно измениться при добавлении текста."
    : repeatNote;

  els.verdictLabel.textContent = labels[0];
  els.verdictLabel.className = `verdict-label ${labels[1]}`;
  els.verdictConf.hidden = false;
  els.meterFill.parentElement.hidden = false;
  els.verdictConf.textContent = `сила ${pctText}`;
  els.meterFill.style.width = `${pct.toFixed(1)}%`;
  // The tooltip states the actual formula; the numbers live on z/p in the table.
  els.verdictConfHelp.dataset.tip =
    `«Сила» считается из z-оценки: шкала линейная, z/6 в процентах — 0% при z = 0, ≈33% при z = 2, ≈67% при z = 4 и 100% при z = 6 (номинальное p ≈ 10⁻⁹).\n\nЕсли данных меньше нормы — от 25 неповторяющихся токенов и не меньше 20 наблюдений для аппроксимации, — процент пропорционально снижается.\n\nЭто не вероятность авторства ИИ. Точные числа — z-оценка и p-значение в таблице ниже.`;

  els.statScheme.textContent = tournament ? "SynthID" : "зелёный список";
  els.statFirstLabel.textContent = tournament ? "взвешенное среднее" : "зелёные токены";
  els.statGreen.textContent = tournament
    ? `${(g * 100).toFixed(1)}% (${T} × ${m})`
    : `${greenCount} / ${T} (${T ? (greenCount / T * 100).toFixed(0) : "0"}%)`;
  els.statMean.parentElement.hidden = !tournament;
  els.statMean.textContent = `${(plainMean * 100).toFixed(1)}%`;
  els.statZ.textContent = T ? zVal.toFixed(2) : "—";
  els.statP.textContent = T ? fmtP(pValue) : "—";
  els.verdictNote.hidden = !note;
  els.verdictNoteHelp.hidden = !note;
  els.verdictNote.textContent = note;
  setVerdictTips({ tournament, z: zVal, pValue, greenCount, T, totalT, gamma: gammaVal, meanG: plainMean, weightedMeanG: g, m, h, ready, reference, expectedSource });
}

function setVerdictTips({ tournament, z, pValue, greenCount, T, totalT, gamma, meanG, weightedMeanG, m, h, ready, reference, expectedSource }) {
  const tip = (id, text) => {
    const help = document.getElementById(id === "verdict-note" ? "verdict-note-help" : `${id}-help`);
    if (help) help.dataset.tip = text;
  };
  const counting = tournament
    ? `Учтено ${T} из ${totalT} проверяемых токенов. Повторяющийся контекст учитывается один раз — как в официальном детекторе synthid-text (compute_context_repetition_mask): повтор не даёт нового свидетельства метки.`
    : `Учтено ${T} из ${totalT} проверяемых токенов. Одинаковый токен после тех же предыдущих токенов учитывается один раз: повтор не даёт нового свидетельства метки.`;
  const baseline = tournament
    ? "доля оценок 1 — около 50%"
    : `доля зелёных токенов — около ${(gamma * 100).toFixed(1)}%`;
  if (tournament) {
    tip("tip-scheme",
      `Выбран детектор турнирной метки с официальным скорингом synthid-text (Google DeepMind): функции weighted_mean_score и mean_score из detector_mean.py. Обучение не требуется.\n\nОн восстанавливает служебные оценки 0 и 1 по ключу и ищет их избыток. Вердикт и шкала определяются статистикой z с учётом объёма данных, а не близостью среднего g к эталону.\n\nРаундов на токен: ${m} (параметр m). Предыдущих токенов в расчёте: ${h} (параметр h). Токен — слово или часть слова. Эти параметры и ключ должны совпадать с генерацией.`
    );
    tip("tip-green",
      `В каждом раунде правило турнира присваивает токену оценку 0 или 1; при генерации 1 даёт преимущество перед 0. Оценки вычисляются из ключа и контекста — в самом тексте их нет.\n\nПоказанная величина — смещение взвешенного среднего g от 50%; значимость этого смещения измеряется отдельно z-оценкой.\n\nВзвешенное среднее: ${(weightedMeanG * 100).toFixed(1)}% — веса раундов от 10 до 1. Простое среднее: ${(meanG * 100).toFixed(1)}% из ${T * m} оценок. Справочный ориентир для помеченного текста при равномерной модели: ${(reference * 100).toFixed(1)}% (${expectedSource === "official" ? "официальная формула" : "синтетическое распределение"}) — не порог и не критерий вердикта.\n\n` + counting
    );
    tip("tip-mean",
      `Простое среднее — официальная функция mean_score из detector_mean.py: среднее всех учтённых g-оценок без весов.\n\nЭто величина смещения, а не мера значимости: значимость учитывает объём данных и измеряется z-оценкой.`
    );
  } else {
    tip("tip-scheme",
      `Выбран детектор зелёного списка. Он ищет варианты продолжения, которым правило метки даёт преимущество. Проверка одинакова для жёсткого и мягкого режимов.\n\nВердикт и шкала определяются статистикой z с учётом объёма данных, а не превышением доли зелёных над эталоном.\n\nОжидаемая доля зелёных без метки: ${(gamma * 100).toFixed(1)}% (параметр γ). Предыдущих токенов в расчёте: ${h} (параметр h). Токен — слово или часть слова. Эти параметры и ключ должны совпадать с генерацией.`
    );
    tip("tip-green",
      `Зелёный токен — фрагмент текста, которому правило метки давало преимущество в этой позиции. Детектор восстанавливает цвет по ключу и предыдущим токенам.\n\nЗелёных: ${greenCount} из ${T} против ${(gamma * 100).toFixed(1)}% ожидаемых без метки; значимость превышения измеряется z-оценкой.\n\n` + counting
    );
    tip("tip-mean", "");
  }
  tip("tip-z",
    `z-оценка — основной показатель статистического свидетельства: превышение сигнала над уровнем без метки, измеренное в стандартных отклонениях нулевой модели. Без метки ожидается: ${baseline}. z около 0 — как без метки; отрицательное z — ниже этого уровня; z = 4 — превышение на четыре стандартных отклонения.` +
    (tournament
      ? `\n\nДля турнира z вычисляется по той же взвешенной сумме, что и Weighted Mean (веса раундов от 10 до 1): ранние раунды несут более сильный сигнал, и вердикт этим пользуется. При равномерных весах формула совпадает с тестом простого среднего.`
      : "") +
    `\n\nПредпосылка расчёта — независимость учтённых наблюдений: каждый g-бит является отдельным псевдослучайным битом хэша (раунды используют разные константы хэша, повторяющиеся контексты исключены маской), а в зелёном списке каждое учтённое положение — испытание Бернулли с долей γ. Поэтому у турнира учитываются T·m наблюдений, а не только T токенов.\n\nВердикты: z ≥ 2 — слабый сигнал, z ≥ 4 — сильный. Пороги номинальные: они не являются гарантированной частотой ложных срабатываний. Для калибровки прогоните детектор по корпусу немаркированных текстов (например, человеческие тексты или исходный код) и посмотрите распределение z.`
  );
  tip("tip-p",
    `p-значение — номинальная вероятность того, что без водяного знака случайно возник бы столь же сильный или более сильный сигнал. Оно вычисляется из z-оценки в предположении нормальности и независимости наблюдений.\n\nЭто не вероятность того, что текст написал человек или ИИ, и не откалиброванная частота ложных срабатываний: точная оценка требует эмпирической проверки на немаркированных текстах. Запись 1e-15 означает единицу, делённую на 10 в пятнадцатой степени.\n\nЧем меньше p, тем необычнее результат при отсутствии метки. На коротких текстах значением p нельзя пользоваться как точной вероятностью.`
  );
  tip("verdict-note",
    (tournament
      ? `В турнире каждый токен — слово или часть слова — получает оценку 0 или 1 в каждом раунде. Оценка 1 даёт преимущество при выборе продолжения; сами цифры в текст не вставляются.\n\nРазличайте три разные величины. Величина смещения: взвешенное среднее ${(weightedMeanG * 100).toFixed(1)}% (веса раундов от 10 до 1), простое среднее ${(meanG * 100).toFixed(1)}% при 50% без метки. Статистическая значимость: z = ${z.toFixed(2)}. Номинальная вероятность ложного срабатывания: p-значение из таблицы. Вероятность того, что текст создан ИИ, детектор не определяет.`
      : `Зелёные токены — слова или части слов, которым правило метки даёт преимущество; здесь их ${T ? (greenCount / T * 100).toFixed(1) : "0"}% при ожидаемых без метки ${(gamma * 100).toFixed(1)}%.\n\nРазличайте три величины: это фактическая величина смещения; статистическая значимость — z = ${z.toFixed(2)}; номинальная вероятность ложного срабатывания — p-значение из таблицы. Вероятность того, что текст создан ИИ, детектор не определяет.`) +
    `\n\n${counting}\n\n` +
    (!ready ? "Пока данных мало: результат может сильно измениться при добавлении текста. " : "") +
    "Вердикт выбирается по номинальным порогам z (≥ 2 — слабый сигнал, ≥ 4 — сильный). Отсутствие сигнала не доказывает, что текст написал человек: метку могли не ставить, изменить текст или проверить с другим ключом."
  );
}

function fmtP(p) {
  if (p < 1e-15) return "< 1e-15";
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

/* ── per-text strength chart: local watermark strength along the text ── */

/** Null-model sd of one token's weighted score (tournament). */
function tournamentTokenSigma(m) {
  const raw = layerWeights(m);
  const sum = raw.reduce((a, b) => a + b, 0);
  let s2 = 0;
  for (const x of raw) {
    const wn = (x * m) / sum; // normalised like weighted_mean_score
    s2 += wn * wn;
  }
  return (0.5 * Math.sqrt(s2)) / m;
}

/**
 * Rolling local strength of the mark along the text. For every sampled
 * position the mean of the per-token scores over a window of ±window/2
 * tokens is computed (masked-out tokens are skipped, so repeated contexts
 * contribute nothing). Each point carries the null-model sigma of its mean
 * so the chart can shade the ±2σ "ordinary noise" band.
 *
 * @param {number[]} scores per-token scores in [0, 1] (weighted g for the
 *   tournament, 0/1 green flags for the green list)
 * @param {{mask?: ArrayLike<number>|null, sigmaPerToken?: number, window?: number|null, maxPoints?: number}} opts
 * @returns {{window: number, points: Array<{t: number, v: number, n: number, sigma: number}>}}
 */
function computeStrengthSeries(scores, { mask = null, sigmaPerToken = 0.5, window = null, maxPoints = 480 } = {}) {
  const total = scores.length;
  if (total < 2) return { window: 1, points: [] };
  const W = Math.max(2, Math.min(window ?? Math.min(100, Math.max(8, Math.round(total / 8))), total));
  const use = mask ?? scores.map(() => 1);
  const pref = new Float64Array(total + 1);
  const cnt = new Uint32Array(total + 1);
  for (let i = 0; i < total; i++) {
    pref[i + 1] = pref[i] + (use[i] ? scores[i] : 0);
    cnt[i + 1] = cnt[i] + (use[i] ? 1 : 0);
  }
  const step = Math.max(1, Math.ceil(total / maxPoints));
  const half = W / 2;
  const points = [];
  for (let i = 0; i < total; i += step) {
    const lo = Math.max(0, Math.floor(i - half));
    const hi = Math.min(total, Math.ceil(i + half));
    const n = cnt[hi] - cnt[lo];
    if (n <= 0) continue;
    points.push({
      t: i,
      v: (pref[hi] - pref[lo]) / n,
      n,
      sigma: sigmaPerToken / Math.sqrt(n),
    });
  }
  return { window: W, points };
}

/** Draws the strength chart into #strength-chart from a detection message. */
function renderStrengthChart(msg) {
  const host = els.strengthChart;
  if (!host) return;
  const tournament = msg.scheme === "tournament";
  const scores = tournament
    ? msg.perTokenWeighted
    : (msg.flags ?? []).map((f) => +f);
  const total = scores ? scores.length : 0;
  if (total < 10) {
    host.hidden = true;
    els.strengthChartPlot.replaceChildren();
    return;
  }
  const mask = tournament ? msg.mask : null;
  const sigmaPerToken = tournament
    ? tournamentTokenSigma(msg.m || 1)
    : Math.sqrt(Math.max(1e-9, (msg.gamma ?? 0.5) * (1 - (msg.gamma ?? 0.5))));
  const { window: W, points } = computeStrengthSeries(scores, { mask, sigmaPerToken });
  if (points.length < 2) {
    host.hidden = true;
    els.strengthChartPlot.replaceChildren();
    return;
  }

  const Wc = 480;
  const Hc = 320;
  const padL = 55;
  const padR = 18;
  const padT = 40;
  const padB = 65;
  const iw = Wc - padL - padR;
  const ih = Hc - padT - padB;
  const yLo = tournament ? 0.5 : (msg.gamma ?? 0.5); // no-mark level
  const yMin = 0.4; // fixed vertical scale: 40–60%
  const yMax = 0.6;
  const x = (t) => padL + (iw * t) / Math.max(1, total - 1);
  const y = (v) => padT + ih * (1 - (Math.min(yMax, Math.max(yMin, v)) - yMin) / (yMax - yMin));

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${Wc} ${Hc}`);
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", `Сигнал метки по ходу текста: горизонтальная ось — позиция токена от 1 до ${total}, вертикальная — ${tournament ? "взвешенное среднее g" : "доля зелёных токенов"} от 40 до 60 процентов.`);
  const el = (name, attrs, text) => {
    const node = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text != null) node.textContent = text;
    svg.appendChild(node);
    return node;
  };

  for (const value of [0.4, 0.45, 0.5, 0.55, 0.6]) {
    const yy = y(value).toFixed(1);
    el("line", { x1: padL, x2: Wc - padR, y1: yy, y2: yy, class: "sc-grid" });
    el("text", { x: padL - 12, y: Number(yy) + 6, "text-anchor": "end", class: "sc-axis-text" }, `${Math.round(value * 100)}%`);
  }
  for (const [position, label, anchor] of [[0, "1", "start"], [(total - 1) / 2, String(Math.round((total + 1) / 2)), "middle"], [total - 1, String(total), "end"]]) {
    const xx = x(position).toFixed(1);
    el("line", { x1: xx, x2: xx, y1: Hc - padB, y2: Hc - padB + 7, class: "sc-axis" });
    el("text", { x: xx, y: Hc - padB + 29, "text-anchor": anchor, class: "sc-axis-text" }, label);
  }
  el("line", { x1: padL, x2: padL, y1: padT, y2: Hc - padB, class: "sc-axis" });
  el("line", { x1: padL, x2: Wc - padR, y1: Hc - padB, y2: Hc - padB, class: "sc-axis" });
  el("text", { x: Wc / 2, y: Hc - 12, "text-anchor": "middle", class: "sc-axis-title" }, "Позиция в тексте (токены)");
  el("text", { x: padL, y: 24, class: "sc-axis-title" }, tournament ? "Взвешенное среднее, %" : "Доля зелёных, %");

  // ±2σ ordinary range around the unmarked baseline.
  let band = "";
  for (const p of points) band += `${x(p.t).toFixed(1)},${y(Math.min(1, yLo + 2 * p.sigma)).toFixed(1)} `;
  for (let i = points.length - 1; i >= 0; i--) {
    const p = points[i];
    band += `${x(p.t).toFixed(1)},${y(Math.max(0, yLo - 2 * p.sigma)).toFixed(1)} `;
  }
  el("polygon", { points: band.trim(), class: "sc-band" });
  el("line", { x1: padL, x2: Wc - padR, y1: y(yLo).toFixed(1), y2: y(yLo).toFixed(1), class: "sc-null" });
  el("polyline", {
    points: points.map((p) => `${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join(" "),
    class: "sc-line",
  });

  // Hover trace lives entirely inside the SVG: a guide line, a dot on the
  // curve and one number — the local window value at that point. Nothing
  // outside the chart is rewritten, so the layout cannot jump on hover.
  const guide = el("line", { x1: 0, x2: 0, y1: padT, y2: Hc - padB, class: "sc-guide", visibility: "hidden" });
  const dot = el("circle", { cx: 0, cy: 0, r: 6, class: "sc-dot", visibility: "hidden" });
  const tip = el("text", { x: 0, y: 0, class: "sc-tip", "text-anchor": "middle", visibility: "hidden" });
  const pick = (evt) => {
    const rect = svg.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, ((evt.clientX - rect.left) / rect.width * Wc - padL) / iw));
    const t = frac * (total - 1);
    let best = points[0];
    for (const p of points) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
    const px = x(best.t);
    const py = y(best.v);
    guide.setAttribute("x1", px.toFixed(1));
    guide.setAttribute("x2", px.toFixed(1));
    guide.setAttribute("visibility", "visible");
    dot.setAttribute("cx", px.toFixed(1));
    dot.setAttribute("cy", py.toFixed(1));
    dot.setAttribute("visibility", "visible");
    tip.textContent = `${(best.v * 100).toFixed(1)}%`;
    tip.setAttribute("x", Math.min(Wc - padR - 36, Math.max(padL + 36, px)).toFixed(1));
    tip.setAttribute("y", Math.max(padT + 16, py - 14).toFixed(1));
    tip.setAttribute("visibility", "visible");
  };
  const reset = () => {
    for (const node of [guide, dot, tip]) node.setAttribute("visibility", "hidden");
  };
  svg.addEventListener("mousemove", pick);
  svg.addEventListener("mouseleave", reset);

  els.strengthChartPlot.replaceChildren(svg);
  els.strengthChartHelp.dataset.tip = tournament
    ? `Взвешенное среднее по ходу текста. Линия — среднее взвешенных g-оценок (веса как в Weighted Mean) в скользящем окне ${W} токенов: та же величина, что «взвешенное среднее» в таблице выше, но посчитанная локально по окну. Пунктир — уровень ${(yLo * 100).toFixed(0)}% без метки, серая полоса — обычный случайный разброс (±2 стандартных отклонения). Участок, где линия устойчиво выше полосы, несёт метку; внутри полосы — как без метки. Наведите курсор, чтобы увидеть значение в конкретном месте.`
    : `Доля зелёных токенов по ходу текста. Линия — доля зелёных в скользящем окне ${W} токенов: та же величина, что «зелёные токены» в таблице выше, но посчитанная локально по окну. Пунктир — ожидаемая без метки доля γ = ${(yLo * 100).toFixed(0)}%, серая полоса — обычный случайный разброс (±2 стандартных отклонения). Участок, где линия устойчиво выше полосы, несёт метку; внутри полосы — как без метки. Наведите курсор, чтобы увидеть значение в конкретном месте.`;
  host.hidden = false;
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
  els.colorHelp.dataset.tip = tournament
    ? "Показывает оценки отдельных токенов — слов или частей слов. В каждом раунде правило турнира даёт токену 0 или 1; единица даёт преимущество при генерации. Зелёный цвет: единиц не меньше половины, красный: меньше половины. Цвет не означает, что слово правильное или ошибочное. Выключение раскраски не меняет проверку."
    : "Показывает, какие токены — слова или части слов — входят в зелёный и красный списки по ключу детектора. Зелёные получают преимущество при генерации с меткой. Отдельное зелёное слово встречается и без метки; важна общая доля. Выключение раскраски не меняет проверку.";
  setLegend(color);
  renderVerdict(msg);
  renderStrengthChart(msg);
}

function renderTokenChips(tokens, originalText = null) {
  const display = originalText === null
    ? { colouredTokens: tokens, remainder: "" }
    : displayParts(tokens, originalText);
  for (const node of [...els.output.childNodes]) {
    if (node !== els.placeholder && node !== els.promptEcho && node !== els.tokens) node.remove();
  }
  els.placeholder.hidden = true;
  els.promptEcho.textContent = "";
  els.tokens.textContent = "";
  if (els.copyOutput) els.copyOutput.disabled = tokens.length === 0;
  setLegend(false);
  const fragment = document.createDocumentFragment();
  for (const tok of display.colouredTokens) {
    const span = document.createElement("span");
    span.className = "tok";
    span.textContent = tok.text;
    span.title = `Токен — фрагмент текста: слово, часть слова или знак препинания. Номер в словаре модели: ${tok.id}.`;
    fragment.appendChild(span);
  }
  if (display.remainder) fragment.appendChild(document.createTextNode(display.remainder));
  els.tokens.appendChild(fragment);
}

els.colorText.addEventListener("change", () => {
  if (lastPaint) paintDetection(lastPaint);
  else els.legend.hidden = true;
});

els.output.addEventListener("pointerdown", () => {
  if (!els.placeholder.hidden) els.placeholder.hidden = true;
});
els.output.addEventListener("beforeinput", (event) => {
  if (!event.cancelable || !event.inputType.startsWith("delete") || !els.tokens.querySelector(".tok")) return;
  const selection = getSelection();
  if (!selection?.rangeCount) return;
  const range = selection.getRangeAt(0);
  if (range.collapsed) return;
  const insideTokens = (node) => node === els.tokens || els.tokens.contains(node);
  if (!insideTokens(range.startContainer) || !insideTokens(range.endContainer)) return;

  // Deleting across thousands of token spans makes contenteditable stall.
  // Convert them to one text node first, keeping the exact selection.
  const offset = (node, position) => {
    const prefix = document.createRange();
    prefix.setStart(els.tokens, 0);
    prefix.setEnd(node, position);
    return prefix.toString().length;
  };
  const start = offset(range.startContainer, range.startOffset);
  const end = offset(range.endContainer, range.endOffset);
  const value = els.tokens.textContent;
  const plain = document.createTextNode(value.slice(0, start) + value.slice(end));
  event.preventDefault();
  els.tokens.replaceChildren(plain);
  const newRange = document.createRange();
  newRange.setStart(plain, start);
  selection.removeAllRanges();
  selection.addRange(newRange);
  outputDirty = true;
  outputEditVersion++;
  els.placeholder.hidden = true;
  if (els.copyOutput) els.copyOutput.disabled = !plain.data.trim();
  updateButtons();
});
els.output.addEventListener("input", () => {
  outputDirty = true;
  outputEditVersion++;
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
  outputEditVersion++;
  updateButtons();
});

function outputTextFromDom() {
  // Walk the DOM instead of innerText: innerText collapses runs of spaces
  // and drops leading indentation, which changes code tokens (e.g. Python).
  let out = "";
  const walk = (node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      out += node.textContent;
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      if (node.hidden || node.id === "output-placeholder") return;
      if (node.tagName === "BR") {
        out += "\n";
        return;
      }
      for (const child of node.childNodes) walk(child);
      if (node.tagName === "DIV" || node.tagName === "P") out += "\n";
    }
  };
  walk(els.output);
  out = out.replace(/\u00a0/g, " ");
  return out.trim() ? out : "";
}

function textToCheck() {
  return outputDirty ? outputTextFromDom() : "";
}

els.detectBtn.addEventListener("click", () => {
  const text = textToCheck();
  if (text) {
    const requestId = ++detectionRequestId;
    pendingTextDetection = { id: requestId, editVersion: outputEditVersion, text };
    setStatus("Проверяю вставленный текст…");
    send({ type: "detectText", text, requestId, params: detectParams() });
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
  setStatus(`Выбрана ${modelInfo(modelId).name}. Для новой модели нажмите «Скачать выбранную модель и запустить»; для скачанной — «Загрузить» в списке кэша.`);
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
const HINT_SHOW_DELAY = 800; // ms a hover must persist before the tooltip appears
let hintAnchor = null;
let hintPending = null; // host whose tooltip is still waiting out the delay
let hintPinned = false;
let hintShowTimer;
let hintHideTimer;
function tipHost(node) {
  const el = node?.closest?.("[data-tip]");
  return el && !el.classList.contains("tip") && el.dataset.tip ? el : null;
}
function placeHint(anchor) {
  const text = anchor.dataset.tip;
  if (!text) return;
  if (hintAnchor?.classList.contains("tip")) hintAnchor.setAttribute("aria-expanded", "false");
  clearTimeout(hintHideTimer);
  clearTimeout(hintShowTimer);
  hintPending = null;
  hintAnchor = anchor;
  if (anchor.classList.contains("tip")) anchor.setAttribute("aria-expanded", "true");
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
  clearTimeout(hintShowTimer);
  hintPending = null;
  hintPop.hidden = true;
  if (hintAnchor?.classList.contains("tip")) hintAnchor.setAttribute("aria-expanded", "false");
  hintAnchor = null;
  hintPinned = false;
}
document.addEventListener("click", (e) => {
  const help = e.target.closest?.(".tip[data-tip]");
  if (help) {
    e.preventDefault();
    e.stopPropagation();
    if (hintPinned && hintAnchor === help) hideHint();
    else {
      hideHint();
      help.setAttribute("aria-controls", "hint-pop");
      placeHint(help);
      hintPinned = true;
    }
  } else if (hintPinned && !hintPop.contains(e.target)) {
    hideHint();
  }
});
document.addEventListener("mouseover", (e) => {
  if (hintPinned) return;
  if (hintPop.contains(e.target)) {
    clearTimeout(hintHideTimer);
    return;
  }
  const host = tipHost(e.target);
  if (!host) return;
  clearTimeout(hintShowTimer);
  if (host === hintAnchor) {
    clearTimeout(hintHideTimer); // still hovering the same host: keep it open
    return;
  }
  if (hintAnchor) {
    placeHint(host); // another tooltip is already open: switch without delay
    return;
  }
  hintPending = host;
  hintShowTimer = setTimeout(() => {
    if (hintPending === host) placeHint(host);
  }, HINT_SHOW_DELAY);
});
document.addEventListener("mouseout", (e) => {
  if (hintPinned) return;
  const host = tipHost(e.target);
  if (!host && !hintPop.contains(e.target)) return;
  if (hintAnchor?.contains(e.relatedTarget) || hintPop.contains(e.relatedTarget)) return;
  // Left before the show delay elapsed: cancel the pending tooltip.
  if (hintPending) {
    clearTimeout(hintShowTimer);
    hintPending = null;
  }
  // Allow crossing the gap into a long tooltip to scroll its text.
  hintHideTimer = setTimeout(hideHint, 300);
});
document.addEventListener("focusin", (e) => {
  const host = e.target.closest?.("[data-tip]");
  if (host?.dataset.tip && !host.classList.contains("tip") && !hintPinned) placeHint(host);
});
document.addEventListener("focusout", () => { if (!hintPinned) hideHint(); });
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") hideHint();
});
window.addEventListener("scroll", (e) => {
  if (e.target === hintPop) return;
  if (hintPinned && hintAnchor?.isConnected) { placeHint(hintAnchor); return; }
  if (hintAnchor?.contains(document.activeElement)) placeHint(hintAnchor);
  else hideHint();
}, true);
window.addEventListener("resize", hideHint);

syncParamVisibility();
refreshModelOptions(lastBackend);
updateButtons();
setStatus("Выберите модель для скачивания или нажмите «Загрузить» рядом с уже скачанной моделью.");
refreshCachedModels();

/* ── page version: deployed commit hash + date in the header ── */
const VERSION_REPO = "Belerafon/LLM_watermark_payground";

async function loadPageVersion() {
  const el = document.getElementById("page-version");
  if (!el) return;
  // version.json is stamped by the deploy workflow with the exact commit
  // being published. It is the only source: the badge states what this
  // page actually is, not where the branch head happens to be. In a local
  // checkout without one the badge simply stays hidden.
  try {
    const r = await fetch("version.json", { cache: "no-cache" });
    if (!r.ok) return;
    const info = await r.json();
    if (!info?.commit) return;
    const date = new Date(info.time);
    if (Number.isNaN(date.getTime())) return;
    el.textContent = `${info.commit} · ${date.toLocaleString("ru-RU", {
      day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit",
    })}`;
    el.href = info.sha
      ? `https://github.com/${VERSION_REPO}/commit/${info.sha}`
      : `https://github.com/${VERSION_REPO}/commits`;
    el.hidden = false;
  } catch { /* missing or offline: keep the badge hidden */ }
}
loadPageVersion();
