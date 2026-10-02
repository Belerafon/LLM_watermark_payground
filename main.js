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
} from "./models.js?v=3";
import { expectedTournamentMean } from "./watermark.js?v=9";

const $ = (id) => document.getElementById(id);

const els = {
  model: $("model"),
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
  const script = modelInfo(meta.modelId).runtime === "v4" ? "worker-v4.js?v=11" : "worker.js?v=12";
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
  setStatus("Запускаю среду…");
  spawnWorker(meta);
  send({ type: "load", modelId: meta.modelId, device: meta.device ?? null, dtype: meta.dtype ?? null });
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
    [b?.adapter?.description || b?.adapter?.vendor, b?.note].filter(Boolean).join(" — ") ||
    (device === "webgpu" ? "WebGPU" : "WebAssembly, single-threaded");
  els.backendBadge.hidden = false;
}

function updateButtons() {
  els.generateBtn.disabled = !modelReady || generating;
  const hasPaste = outputDirty;
  els.detectBtn.disabled = generating || !modelReady || (!haveGeneration && !hasPaste);
  els.stopBtn.hidden = !generating;
  els.model.disabled = generating;
}

/** Rewrite the dropdown labels with the sizes for the active backend/dtype. */
function refreshModelOptions(backend) {
  for (const opt of els.model.options) {
    const m = modelInfo(opt.value);
    const dtype = m.webgpuOnly ? "q4f16" : backend?.dtype;
    const size = dtype ? sizeMB(opt.value, dtype) : null;
    const parts = [];
    if (size) parts.push(`~${formatMB(size).replace(" GB", " ГБ").replace(" MB", " МБ")}`);
    if (m.webgpuOnly) parts.push("только WebGPU");
    opt.textContent = parts.length ? `${m.name} (${parts.join(", ")})` : m.name;
    opt.disabled = !!m.webgpuOnly && !!backend && backend.device !== "webgpu";
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
  none: "Метки нет. На каждом шаге модель составляет список продолжений и вытягивает одно слово: чаще то, которое сама считает уместным.",
  hard: "Словарь тайно делится на зелёный и красный списки. Красные слова запрещены совсем, модель обязана взять зелёное. Текст остаётся связным, но выбор уже не тот, что она хотела.",
  soft: "Те же списки, но красные не запрещены. Зелёным чуть поднимают шанс, поэтому они выпадают чаще, а не всегда. Текст почти как без метки.",
  tournament: "Без метки модель на каждом шаге составляет список продолжений и выбирает одно: чаще то, которое сама считает уместным.\n\nSynthID этот список не подменяет и в текст ничего секретного не дописывает. Она меняет только розыгрыш.\n\nИз того же списка вытягивают не одно слово, а пачку кандидатов. Все они правдоподобны, просто одни вероятнее других. Дальше их сводят в турнир на вылет. У каждого кандидата есть секретная метка 0 или 1. Она считается из ключа и предыдущего слова, не из смысла. В раунде остаётся тот, у кого 1. Раундов несколько — это число m. В текст попадает только последний победитель.\n\nЧитатель видит обычную фразу: победитель и так был вариантом модели. Но среди одинаково уместных слов чуть чаще остаются те, кому секретные метки благоприятствуют.",
};

const DETECT_HINT = {
  none: "Проверка ищет зелёный список, хотя метку не ставили. На обычном тексте сигнала быть не должно. Ключ и h должны совпасть с тем, что стояло при генерации, иначе проверка считает другую метку.",
  hard: "Проверка модель не запускает и её шансы не знает. Она заново делит словарь на зелёный и красный по ключу и предыдущему слову и считает, не слишком ли много зелёных. Ключ и h должны совпасть с генерацией.",
  soft: "Проверка модель не запускает и её шансы не знает. Она заново делит словарь на зелёный и красный по ключу и предыдущему слову и считает, не слишком ли много зелёных. Ключ и h должны совпасть с генерацией.",
  tournament: "Проверка модель не запускает и её шансы не знает. Она заново считает те же метки 0 и 1 для уже написанных слов, по ключу и предыдущему слову. Без водяного знака их примерно поровну. Если текст помечен, единиц заметно больше половины. Ключ и h должны совпасть с генерацией.",
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
    `Эта кнопка проверяет алгоритмом «${name}». Это не режим генерации слева: тот только ставит метку. ` +
    "Числа h, m, γ и красный список берутся из панели слева, ключ — из поля выше.\n\n" +
    (DETECT_HINT[mode] ?? "");
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
      span.title = `токен ${msg.id}`;
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
        firstFailure: { phase: failure.phase, kind: failure.kind, raw: failure.raw, dtype: failure.dtype, elapsedMs: failure.elapsedMs, bytes: failure.bytes },
      });
      return;
    }

    retireWorker();
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
      startLoad({ modelId: a.modelId, device: a.backend?.device ?? a.device, dtype: forcedDtype, keepOutput: true });
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
      return (
        `${name} не влезла в память браузера на этапе «${where}»${on}. ` +
        `На загрузку нужно ≈${formatMB(need)} памяти WebAssembly (граф собирается, пока веса ещё лежат в памяти)` +
        (heap
          ? `, а этот браузер смог вырастить кучу только до ≈${formatMB(heap)}.`
          : ". У WebAssembly потолок 4 ГБ, на телефонах ещё меньше.") +
        " Эта модель и так самая большая, которую сюда можно поставить."
      );
    case "network":
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

function renderVerdict({ scheme, z, pValue, greenCount, T, gamma, meanG, m, h, expectedG }) {
  els.verdict.hidden = false;
  const tournament = scheme === "tournament";
  const ceiling = tournament ? (Number(expectedG) || expectedTournamentMean(m)) : 1;
  const floor = tournament ? 0.5 : gamma;
  const budget = Math.max(1e-4, ceiling - floor);
  const g = Number.isFinite(meanG) ? meanG : 0.5;
  // Short text: the sample mean wanders above the long-run ceiling. Don't score that as a full mark.
  const checks = Math.max(1, (T || 0) * (m || 1));
  const need = Math.ceil(1 / (budget * budget));
  const ready = !tournament || checks >= need;
  const raw = tournament
    ? Math.min(1, Math.max(0, (g - 0.5) / budget)) * Math.min(1, checks / need)
    : ((T ? greenCount / T : gamma) - gamma) / Math.max(1e-9, 1 - gamma);
  const pct = Math.max(0, Math.min(100, raw * 100));
  const pctText = pct >= 99.5 ? "100%" : pct < 10 ? `${pct.toFixed(1)}%` : `${pct.toFixed(0)}%`;

  let label, cls, note;
  if (pct >= 70) {
    label = "Водяной знак найден";
    cls = "pos";
  } else if (pct >= 30) {
    label = "Слабый след";
    cls = "mid";
  } else {
    label = "Водяной знак не найден";
    cls = "neg";
  }
  if (tournament) {
    note = ready
      ? `Средний g ${g.toFixed(3)}. Предел при m=${m} — ${ceiling.toFixed(3)}. От предела набрано ${pctText}.`
      : `Средний g ${g.toFixed(3)}. Токенов пока ${T}, среднее ещё скачет.`;
  } else {
    const rate = T ? greenCount / T : 0;
    note = `Зелёных ${(rate * 100).toFixed(0)}% при честных ${(gamma * 100).toFixed(0)}%. Процент — доля пути от γ до 100%.`;
  }

  els.verdictLabel.textContent = label;
  els.verdictLabel.className = `verdict-label ${cls}`;
  els.verdictConf.hidden = false;
  els.meterFill.parentElement.hidden = false;
  els.verdictConf.textContent = `метка ${pctText}`;
  els.meterFill.style.width = `${pct.toFixed(1)}%`;
  els.verdictConf.dataset.tip = tournament
    ? `Процент — какая доля пути от 0.500 до предела набрана. Предел считает детектор для текущего m.\n\n` +
      `Сейчас предел ${ceiling.toFixed(3)}. Смените число слоёв и проверьте снова — предел пересчитается.\n\n` +
      `Подпись по той же шкале. От 70% — «найден». От 30% до 70% — «слабый след». Ниже — «не найден».`
    : `Процент — насколько доля зелёных ушла от честных γ к 100%.`;

  els.statScheme.textContent = tournament ? `турнир (m=${m}, h=${h})` : `зелёный список (γ=${gamma}, h=${h})`;
  if (tournament) {
    els.statFirstLabel.textContent = "средний g";
    els.statGreen.textContent = `${meanG.toFixed(3)} на ${T} × ${m}`;
  } else {
    els.statFirstLabel.textContent = "зелёные токены";
    els.statGreen.textContent = `${greenCount} / ${T} (${((greenCount / T) * 100).toFixed(0)}%)`;
  }
  els.statZ.textContent = z.toFixed(2);
  els.statP.textContent = fmtP(pValue);
  els.verdictNote.textContent = note;
  setVerdictTips({ tournament, z, pValue, greenCount, T, gamma, meanG, m, h, note });
}

function setVerdictTips({ tournament, z, pValue, greenCount, T, gamma, meanG, m, h, note }) {
  const pText = fmtP(pValue);
  const tip = (id, text) => {
    const el = document.getElementById(id);
    if (el) el.dataset.tip = text;
  };
  if (tournament) {
    const checks = T * m;
    tip("tip-scheme",
      `Каким способом пересчитывается метка. Сейчас это турнир SynthID, не зелёный список.\n\n` +
      `m=${m} — сколько раундов на вылет у каждого слова. h=${h} — сколько предыдущих слов входит в секрет. h=1 значит метка зависит только от предыдущего токена и ключа.\n\n` +
      `Это не оценка качества текста, а название метода. Смените алгоритм сверху — строка сменится.`
    );
    tip("tip-green",
      `Средний g — доля секретных единиц.\n\n` +
      `У каждого написанного слова и каждого раунда есть метка 0 или 1. Она считается из ключа и предыдущего слова, не из смысла. Среднее ${meanG.toFixed(3)} значит, что единица выпала в ${(meanG * 100).toFixed(1)}% проверок.\n\n` +
      `${T} — сколько токенов проверено. ${m} — раундов на каждый. Вместе ${T}×${m} = ${checks} проверок.\n\n` +
      `Без водяного знака среднее около 0.500. Предел для текущего m считает детектор. Процент сверху — какая доля пути до этого предела набрана.`
    );
  } else {
    const pct = T ? ((greenCount / T) * 100).toFixed(0) : "0";
    tip("tip-scheme",
      `Каким способом пересчитывается метка. Сейчас это зелёный и красный списки, не турнир.\n\n` +
      `γ=${gamma} — какая доля словаря красится в зелёный. h=${h} — сколько предыдущих слов входит в секрет.\n\n` +
      `Это не оценка качества текста. Смените алгоритм сверху — строка сменится.`
    );
    tip("tip-green",
      `Сколько написанных токенов попало в зелёный список.\n\n` +
      `${greenCount} из ${T} — это ${pct}%. Зелёный список каждый раз заново считается из ключа и предыдущего слова.\n\n` +
      `Без метки зелёных должно быть около γ=${gamma}, то есть примерно ${Math.round((gamma ?? 0.5) * 100)}%. Избыток сверх этого и есть след водяного знака.`
    );
  }
  tip("tip-z",
    `Насколько это странно для обычного текста, в шагах обычного разброса.\n\n` +
    `0 — ровно как случайность. 2 — уже редко. 4 — порог из статей: почти наверняка метка. Сейчас ${z.toFixed(2)}.\n\n` +
    `Считается так: сдвиг среднего от честных 0.5 делят на обычный разброс. Чем больше токенов, тем меньший сдвиг уже даёт большой z. Отрицательный z значил бы «единиц меньше, чем у честной монетки».`
  );
  tip("tip-p",
    `Вероятность увидеть такой результат, если водяного знака не было и метки 0/1 выпадали честно.\n\n` +
    `Сейчас ${pText}. Запись 1e-15 — это 0.000000000000001, меньше одного шанса на квадриллион.\n\n` +
    `Маленькое p — не «текст плохой» и не «модель ошиблась». Это «так повезти без метки почти нельзя».`
  );
  tip("verdict-note",
    `Эта фраза пересказывает z и p обычными словами.\n\n` +
    `«Без метки» — если бы единицы и нули выпадали честно, примерно поровну.\n\n` +
    `«Случайно с вероятностью ${pText}» — это p-значение.\n\n` +
    `«Порог z > 4» — в статьях считают, что z больше 4 уже достаточно, чтобы сказать: текст помечен. Сейчас z = ${z.toFixed(2)}.\n\n` +
    note
  );
}

function fmtP(p) {
  if (p < 1e-15) return "< 1e-15";
  if (p < 0.001) return p.toExponential(1);
  return p.toFixed(3);
}

/* ── user actions ── */

els.generateBtn.addEventListener("click", () => {
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

  send({ type: "generate", params: p });
});

els.stopBtn.addEventListener("click", () => send({ type: "interrupt" }));

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
    chips[i].title = tournament && color && known ? `${base} · средний g = ${msg.perTokenScore[flag].toFixed(2)}` : base;
  }
  els.legendHi.textContent = tournament ? "g ≥ 0.5" : "зелёный список";
  els.legendLo.textContent = tournament ? "g < 0.5" : "красный список";
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
    span.title = `токен ${tok.id}`;
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
  if (WEBGPU_ONLY.has(modelId) && forcedDevice !== "webgpu" && lastBackend && lastBackend.device !== "webgpu") {
    retireWorker();
    modelReady = false;
    haveGeneration = false;
    els.backendBadge.hidden = true;
    updateButtons();
    showNote("");
    hideDiagnostics();
    setStatus(`${modelInfo(modelId).name} нужна WebGPU, а в этом браузере её нет (сейчас WASM). Возьмите Chrome или Edge.`, "error");
    return;
  }
  startLoad({ modelId, device: forcedDevice, dtype: forcedDtype });
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
  hintPop.hidden = true;
}
document.addEventListener("mouseover", (e) => {
  const host = tipHost(e.target);
  if (!host) return;
  placeHint(host);
});
document.addEventListener("mouseout", (e) => {
  const host = tipHost(e.target);
  if (!host) return;
  if (host.contains(e.relatedTarget)) return;
  hideHint();
});
document.addEventListener("focusin", (e) => {
  const host = e.target.closest?.("[data-tip]");
  if (host?.dataset.tip) placeHint(host);
});
document.addEventListener("focusout", hideHint);
window.addEventListener("scroll", hideHint, true);

syncParamVisibility();
refreshModelOptions(lastBackend);
els.statusText.title = `build ${APP_VERSION}`;
updateButtons();
startLoad({ modelId: els.model.value, device: forcedDevice, dtype: forcedDtype });
