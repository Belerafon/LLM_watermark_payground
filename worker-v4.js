/**
 * Web worker for models that need Transformers.js v4 (Gemma 4).
 * Qwen3-4B stays on worker.js / Transformers.js 3.7.2.
 *
 * Gemma 4 E4B is loaded as text-only: Gemma4ForCausalLM makes the library skip
 * the vision and audio encoders. Watermarking still goes through logits_processor.
 */
import {
  AutoProcessor,
  Gemma4ForCausalLM,
  TextStreamer,
  LogitsProcessor,
  LogitsProcessorList,
  InterruptableStoppingCriteria,
  env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0/dist/transformers.min.js";
import { seedFromContext, isGreen, detect, keyToSeed, tournamentSample, detectTournament, sampleMultinomial, seededRng } from "./watermark.js?v=8";
import { APP_VERSION, WEBGPU_ONLY, modelInfo, dtypeFor, sizeMB, formatMB, classifyError, describeAdapter, createLog } from "./models.js?v=3";

// Local folder if present (start.bat). Otherwise Hugging Face Hub, so GitHub Pages
// does not need the multi-GB weights in the repo.
// WASM stays next to this file; v4 would otherwise fetch it from jsDelivr.
env.allowLocalModels = true;
env.allowRemoteModels = true;
env.localModelPath = "/models/";
env.useBrowserCache = true;
const wasmBase = new URL("./vendor4/", import.meta.url);
env.backends.onnx.wasm.wasmPaths = {
  // 4.3 uses the asyncify build even for WebGPU (jsep is not the default).
  mjs: new URL("ort-wasm-simd-threaded.asyncify.mjs", wasmBase).href,
  wasm: new URL("ort-wasm-simd-threaded.asyncify.wasm", wasmBase).href,
};

let processor = null;
let tokenizer = null;
let model = null;
let modelId = null;
let device = null;
let dtype = null;
let phase = "starting";
let lastGen = null;

const log = createLog(60);
const stoppingCriteria = new InterruptableStoppingCriteria();
const post = (msg) => self.postMessage(msg);

function setPhase(p) {
  if (phase === p) return;
  phase = p;
  log.push("phase", p);
  post({ type: "phase", phase: p });
}

function fail(failedPhase, err) {
  const c = classifyError(err, device);
  log.push("error", { phase: failedPhase, kind: c.kind, raw: c.raw.slice(0, 300) });
  post({ type: "error", phase: failedPhase, kind: c.kind, raw: c.raw, name: c.name, stack: c.stack, modelId, device, dtype, log: log.entries() });
}

self.addEventListener("error", (e) => {
  e.preventDefault();
  fail(phase, e.error ?? e.message);
});
self.addEventListener("unhandledrejection", (e) => {
  e.preventDefault();
  fail(phase, e.reason);
});

/** v4 hands generate() a float32 tensor; some processors index it as logits[i]. */
function logitView(logits) {
  if (logits?.data && logits.dims) {
    const vocab = logits.dims.at(-1);
    const offset = Math.max(0, logits.data.length - vocab);
    return { data: logits.data, vocab, offset };
  }
  const row = logits?.[0] ?? logits;
  const data = row.data;
  const vocab = row.dims?.at(-1) ?? data.length;
  return { data, vocab, offset: 0 };
}

/**
 * Same watermark order as worker.js. input_ids in v4 are bigint[][] from tolist().
 */
class WatermarkProcessor extends LogitsProcessor {
  constructor({ mode, gamma, delta, h, m, forcedRed, temperature, topK, topP, keySeed, rng, sampleSelf }) {
    super();
    this.mode = mode;
    this.gamma = gamma;
    this.delta = delta;
    this.h = h;
    this.m = m;
    this.forcedRed = forcedRed;
    this.temperature = temperature;
    this.topK = topK;
    this.topP = topP;
    this.keySeed = keySeed;
    this.rng = rng;
    this.sampleSelf = sampleSelf;
    this.probs = null;
  }

  _call(input_ids, logits) {
    const { data, vocab, offset } = logitView(logits);
    const at = (t) => data[offset + t];
    const set = (t, v) => {
      data[offset + t] = v;
    };

    let seed = 0;
    if (this.mode !== "none") {
      const ids = input_ids[0];
      const ctx = ids.slice(Math.max(0, ids.length - this.h)).map(Number);
      seed = seedFromContext(ctx, this.keySeed);
    }

    if (this.mode === "hard") {
      for (let t = 0; t < vocab; t++) {
        if (!isGreen(seed, t, this.gamma, this.forcedRed)) set(t, -Infinity);
      }
    } else if (this.mode === "soft") {
      for (let t = 0; t < vocab; t++) {
        if (isGreen(seed, t, this.gamma, this.forcedRed)) set(t, at(t) + this.delta);
      }
    }

    if (this.temperature > 0 && this.temperature !== 1) {
      const inv = 1 / this.temperature;
      for (let i = 0; i < vocab; i++) set(i, at(i) * inv);
    }

    const probs = this.softmax(at, vocab);
    this.truncate(set, probs, vocab);

    if (this.mode === "tournament") {
      const winner = tournamentSample(probs, seed, this.m, this.forcedRed, this.rng);
      for (let t = 0; t < vocab; t++) set(t, t === winner ? 0 : -Infinity);
    } else if (this.sampleSelf) {
      const winner = sampleMultinomial(probs, 1, this.rng)[0];
      for (let t = 0; t < vocab; t++) set(t, t === winner ? 0 : -Infinity);
    }
    return logits;
  }

  softmax(at, vocab) {
    if (!this.probs || this.probs.length !== vocab) this.probs = new Float64Array(vocab);
    const probs = this.probs;
    let max = -Infinity;
    for (let t = 0; t < vocab; t++) {
      const v = at(t);
      if (v > max) max = v;
    }
    let sum = 0;
    for (let t = 0; t < vocab; t++) {
      const v = at(t);
      const e = v === -Infinity ? 0 : Math.exp(v - max);
      probs[t] = e;
      sum += e;
    }
    for (let t = 0; t < vocab; t++) probs[t] /= sum;
    return probs;
  }

  truncate(set, probs, vocab) {
    const k = this.topK;
    const p = this.topP;
    const useK = k > 0 && k < vocab;
    const useP = p > 0 && p < 1;
    if (!useK && !useP) return;

    const sorted = Float64Array.from(probs).sort();
    let threshold = 0;
    if (useK) threshold = sorted[vocab - k];
    if (useP) {
      let cum = 0;
      for (let i = vocab - 1; i >= 0; i--) {
        cum += sorted[i];
        if (cum >= p) {
          threshold = Math.max(threshold, sorted[i]);
          break;
        }
      }
    }

    let sum = 0;
    for (let t = 0; t < vocab; t++) {
      if (probs[t] < threshold || probs[t] === 0) {
        set(t, -Infinity);
        probs[t] = 0;
      } else {
        sum += probs[t];
      }
    }
    if (sum > 0) for (let t = 0; t < vocab; t++) probs[t] /= sum;
  }
}

async function pickBackend(forced) {
  const out = { device: "wasm", f16: false, note: "", adapter: null };
  if (forced === "wasm") {
    out.note = "WASM forced via ?device=wasm";
    return out;
  }
  if (!self.navigator?.gpu) {
    if (forced === "webgpu") {
      throw Object.assign(new Error("WebGPU forced via ?device=webgpu, but navigator.gpu is not available in a worker in this browser."), { kind: "webgpu" });
    }
    out.note = "navigator.gpu is not available in this worker → WASM";
    return out;
  }
  try {
    const adapter = await self.navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("requestAdapter() returned null (WebGPU disabled or GPU blocklisted)");
    out.adapter = describeAdapter(adapter);
    out.f16 = out.adapter.shaderF16;
    if (out.adapter.isFallbackAdapter && forced !== "webgpu") {
      throw new Error("only a software (fallback) WebGPU adapter is available");
    }
    const dev = await adapter.requestDevice();
    dev.destroy?.();
    out.device = "webgpu";
    if (!out.f16) out.note = "adapter lacks shader-f16; Gemma 4 E4B is bundled only as q4f16";
    return out;
  } catch (e) {
    if (forced === "webgpu") {
      throw Object.assign(new Error(`WebGPU forced via ?device=webgpu, but it is not usable here: ${e?.message ?? e}`), { kind: "webgpu" });
    }
    out.device = "wasm";
    out.note = `WebGPU unavailable (${e?.message ?? e}) → WASM`;
    return out;
  }
}

function seqLen(inputIds) {
  if (inputIds?.dims) return inputIds.dims.at(-1);
  if (Array.isArray(inputIds?.[0])) return inputIds[0].length;
  return inputIds?.length ?? 0;
}

async function loadModel(msg) {
  modelId = msg.modelId;
  const info = modelInfo(modelId);
  try {
    setPhase("backend");
    const backend = await pickBackend(msg.device);
    device = backend.device;
    dtype = dtypeFor(device, backend.f16, msg.dtype);
    log.push("backend", { device, dtype, note: backend.note, adapter: backend.adapter, wasm: env.backends.onnx.wasm.wasmPaths });
    post({
      type: "backend",
      device,
      dtype,
      f16: backend.f16,
      note: backend.note,
      adapter: backend.adapter,
      transformers: env.version,
      wasm: { numThreads: env.backends?.onnx?.wasm?.numThreads, proxy: env.backends?.onnx?.wasm?.proxy },
    });

    if (device !== "webgpu" && WEBGPU_ONLY.has(modelId)) {
      throw Object.assign(
        new Error(
          `${info.name} нужна WebGPU: ${formatMB(sizeMB(modelId, "q4f16"))} весов не влезают в 32-битный WASM. ` +
            "Откройте страницу в Chrome или Edge."
        ),
        { kind: "unsupported" }
      );
    }
    if (dtype !== "q4f16") {
      throw Object.assign(
        new Error(`${info.name} лежит только в q4f16. Нужна видеокарта с shader-f16.`),
        { kind: "unsupported" }
      );
    }

    const size = sizeMB(modelId, dtype);
    post({
      type: "status",
      text: `Загрузка ${info.name} (${device}, ${dtype}${size ? `, ~${formatMB(size)}` : ""}, только текст)…`,
    });

    setPhase("download");
    processor = await AutoProcessor.from_pretrained(modelId);
    tokenizer = processor.tokenizer;
    log.push("tokenizer");

    const isWeightFile = (f) => /\.onnx(_data(_\d+)?)?$/.test(f ?? "");
    const files = new Map();
    let downloading = false;
    let sawOverall = false;
    let shownPct = 0;
    // One scale for every weight file. Per-file ratios hit 100% on the small
    // graph, then the next multi-GB shard starts the bar over.
    const expectedBytes = Math.max(1, (size ?? 4905) * 1e6);
    const reportLoad = (pct, loaded, total) => {
      const next = Math.min(99, Math.max(shownPct, pct));
      shownPct = next;
      post({
        type: "progress",
        files: files.size || 1,
        progress: next,
        loaded,
        total: total || expectedBytes,
      });
    };
    // Gemma4ForCausalLM + a ConditionalGeneration config skips vision/audio sessions.
    model = await Gemma4ForCausalLM.from_pretrained(modelId, {
      device,
      dtype,
      progress_callback: (p) => {
        if (p.status === "progress_total") {
          sawOverall = true;
          const loaded = p.loaded ?? 0;
          const total = p.total || expectedBytes;
          reportLoad(p.progress ?? (total ? (100 * loaded) / total : 0), loaded, total);
          return;
        }
        if (!isWeightFile(p.file)) return;
        const name = p.file.split("/").pop();
        if (p.status === "initiate") {
          files.set(p.file, { loaded: 0, total: 0, done: false });
          log.push("file", { name, status: "initiate" });
          setPhase("download");
        } else if (p.status === "progress") {
          downloading = true;
          files.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0, done: false });
          if (!sawOverall) {
            let loaded = 0;
            for (const f of files.values()) loaded += f.loaded;
            reportLoad((100 * loaded) / expectedBytes, loaded, expectedBytes);
          }
        } else if (p.status === "done") {
          const f = files.get(p.file) ?? { loaded: 0, total: 0 };
          f.done = true;
          f.loaded = Math.max(f.loaded, f.total);
          files.set(p.file, f);
          log.push("file", { name, status: "done", mb: Math.round(Math.max(f.total, f.loaded) / 1e6), cached: !downloading });
          if ([...files.values()].every((x) => x.done)) {
            setPhase("compile");
            post({ type: "progress", files: files.size, progress: 100, loaded: expectedBytes, total: expectedBytes });
            post({ type: "status", text: "Компиляция Gemma 4 E4B (может занять несколько минут)…" });
          }
        }
      },
    });
    setPhase("ready");
    post({ type: "ready", device, dtype, modelId, log: log.entries() });
  } catch (err) {
    fail(phase, err);
  }
}

function encodeIds(text) {
  if (typeof tokenizer.encode === "function") {
    return Array.from(tokenizer.encode(text, { add_special_tokens: false }), Number);
  }
  const out = tokenizer(text, { add_special_tokens: false });
  const ids = out.input_ids?.tolist?.() ?? out.input_ids;
  const flat = Array.isArray(ids?.[0]) ? ids[0] : ids;
  return Array.from(flat, Number);
}

function buildForcedRedSet(redWords) {
  const set = new Set();
  if (!tokenizer || !redWords) return set;
  const words = redWords.split(/[,\n]+/).map((w) => w.trim()).filter(Boolean);
  for (const w of words) {
    const lower = w.toLowerCase();
    const cap = lower.charAt(0).toUpperCase() + lower.slice(1);
    const forms = new Set([w, lower, cap, w.toUpperCase()]);
    for (const form of [...forms]) forms.add(" " + form);
    for (const form of forms) {
      for (const id of encodeIds(form)) set.add(id);
    }
  }
  return set;
}

function stripTrailingSpecial(ids) {
  let end = ids.length;
  while (end > 0 && tokenizer.decode([ids[end - 1]], { skip_special_tokens: true }) === "") {
    end--;
  }
  return ids.slice(0, end);
}

async function generate(p) {
  if (!model || !tokenizer) {
    post({ type: "error", phase: "generate", kind: "other", raw: "Модель ещё не загружена.", modelId, device, dtype, log: log.entries() });
    return;
  }
  setPhase("generate");
  try {
    const forcedRed = buildForcedRedSet(p.redWords);
    const prompt = p.prompt.trim();

    const inputs = await processor.apply_chat_template([{ role: "user", content: prompt }], {
      add_generation_prompt: true,
      tokenize: true,
      return_dict: true,
      enable_thinking: false,
    });
    const maxInput = Math.max(128, Math.round(p.maxInputTokens || 16384));
    const fullLen = seqLen(inputs.input_ids);
    if (fullLen > maxInput) {
      const start = fullLen - maxInput;
      for (const key of Object.keys(inputs)) {
        const t = inputs[key];
        if (t?.dims && t.dims.at(-1) === fullLen && typeof t.slice === "function") {
          inputs[key] = t.slice(null, [start, null]);
        }
      }
      post({ type: "status", text: `Промпт длиннее ${maxInput} токенов — оставлено окончание.` });
    }
    const promptLen = seqLen(inputs.input_ids);

    const greedy = p.temperature <= 0;
    const hasSeed = p.seed !== null && p.seed !== undefined;
    const rng = hasSeed ? seededRng(Math.round(p.seed) | 0) : Math.random;
    const processors = new LogitsProcessorList();
    processors.push(
      new WatermarkProcessor({
        mode: p.mode,
        gamma: p.gamma,
        delta: p.delta,
        h: p.h,
        m: p.m,
        forcedRed,
        temperature: greedy ? 1 : p.temperature,
        topK: p.topK,
        topP: p.topP,
        keySeed: keyToSeed(p.key),
        rng,
        sampleSelf: hasSeed && !greedy,
      })
    );

    const promptIds = inputs.input_ids.tolist()[0].map(Number);
    const liveIds = [];
    const LIVE_EVERY = 16;
    const emitLive = () => {
      if (!liveIds.length) return;
      const ids = promptIds.concat(liveIds);
      lastGen = { ids, promptLen: promptIds.length };
      post({ type: "live", ...scoreGeneration(ids, promptIds.length, p) });
    };
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
      callback_function: () => {},
      token_callback_function: (ids) => {
        const list = Array.isArray(ids) || ArrayBuffer.isView(ids) ? Array.from(ids) : [ids];
        for (const id of list) {
          const n = Number(id);
          liveIds.push(n);
          const text = tokenizer.decode([n], { skip_special_tokens: false });
          post({ type: "token", id: n, text });
        }
        if (liveIds.length % LIVE_EVERY === 0) emitLive();
      },
    });

    stoppingCriteria.reset();
    // Library temperature/top_k would run as well; keep them neutral and do both here.
    const output = await model.generate({
      ...inputs,
      max_new_tokens: p.maxNewTokens,
      do_sample: !greedy,
      top_k: 0,
      temperature: 1,
      logits_processor: processors,
      stopping_criteria: stoppingCriteria,
      streamer,
    });

    const sequences = output.sequences ?? output;
    const allIds = stripTrailingSpecial(sequences.tolist()[0].map(Number));
    const genIds = allIds.slice(promptLen);
    lastGen = { ids: allIds, promptLen };
    const detection = genIds.length ? scoreGeneration(allIds, promptLen, p) : null;

    post({
      type: "generated",
      promptText: tokenizer.decode(allIds.slice(0, promptLen), { skip_special_tokens: true }),
      tokens: genIds.map((id) => ({ id, text: tokenizer.decode([id], { skip_special_tokens: false }) })),
      interrupted: stoppingCriteria.interrupted,
      detection,
    });
  } catch (err) {
    fail("generate", err);
  } finally {
    setPhase("ready");
  }
}

function scoreGeneration(ids, promptLen, p) {
  const forcedRed = buildForcedRedSet(p.redWords);
  const keySeed = keyToSeed(p.key);
  if (p.mode === "tournament") {
    const r = detectTournament(ids, promptLen, { m: p.m, h: p.h, forcedRed, keySeed });
    return {
      scheme: "tournament",
      flags: r.flags,
      perTokenScore: r.perTokenScore,
      meanG: r.meanG,
      T: r.T,
      m: r.m,
      h: p.h,
      z: r.z,
      pValue: r.pValue,
    };
  }
  const result = detect(ids, promptLen, { gamma: p.gamma, h: p.h, forcedRed, keySeed });
  return {
    scheme: "greenlist",
    flags: result.flags,
    greenCount: result.greenCount,
    T: result.T,
    z: result.z,
    pValue: result.pValue,
    gamma: p.gamma,
    h: p.h,
  };
}

function tokenChips(ids) {
  return ids.map((id) => ({ id, text: tokenizer.decode([id], { skip_special_tokens: false }) }));
}

function detectPasted(text, p) {
  if (!tokenizer) {
    post({ type: "error", phase: "detect", kind: "other", raw: "Модель ещё не загружена.", modelId, device, dtype, log: log.entries() });
    return;
  }
  const trimmed = String(text ?? "").trim();
  if (!trimmed) {
    post({ type: "error", phase: "detect", kind: "other", raw: "Сначала вставьте текст.", modelId, device, dtype, log: log.entries() });
    return;
  }
  setPhase("detect");
  try {
    let ids = encodeIds(trimmed);
    const cap = 2000;
    let note = "";
    if (!ids.length) {
      post({ type: "error", phase: "detect", kind: "other", raw: "Токенизатор не нашёл ни одного токена.", modelId, device, dtype, log: log.entries() });
      setPhase("ready");
      return;
    }
    if (ids.length > cap) {
      ids = ids.slice(0, cap);
      note = ` Проверены первые ${cap} токенов.`;
    }
    const h = Math.max(1, Math.round(p.h || 1));
    const promptLen = Math.min(h, Math.max(0, ids.length - 1));
    lastGen = { ids, promptLen };
    runDetect(p, { tokens: tokenChips(ids.slice(promptLen)), pasted: true, note });
  } catch (err) {
    fail("detect", err);
    setPhase("ready");
  }
}

function runDetect(p, extra = null) {
  if (!lastGen) {
    post({ type: "error", phase: "detect", kind: "other", raw: "Пока нечего проверять. Сначала сгенерируйте текст.", modelId, device, dtype, log: log.entries() });
    return;
  }
  setPhase("detect");
  try {
    const forcedRed = buildForcedRedSet(p.redWords);
    if (p.scheme === "tournament") {
      const r = detectTournament(lastGen.ids, lastGen.promptLen, {
        m: p.m,
        h: p.h,
        forcedRed,
        keySeed: keyToSeed(p.key),
      });
      post({
        type: "detected",
        scheme: "tournament",
        flags: r.flags,
        perTokenScore: r.perTokenScore,
        meanG: r.meanG,
        T: r.T,
        m: r.m,
        h: p.h,
        z: r.z,
        pValue: r.pValue,
        ...(extra ?? {}),
      });
      return;
    }
    const result = detect(lastGen.ids, lastGen.promptLen, {
      gamma: p.gamma,
      h: p.h,
      forcedRed,
      keySeed: keyToSeed(p.key),
    });
    post({
      type: "detected",
      scheme: "greenlist",
      flags: result.flags,
      greenCount: result.greenCount,
      T: result.T,
      z: result.z,
      pValue: result.pValue,
      gamma: p.gamma,
      h: p.h,
      ...(extra ?? {}),
    });
  } catch (err) {
    fail("detect", err);
  } finally {
    setPhase("ready");
  }
}

self.onmessage = async (e) => {
  const msg = e.data;
  switch (msg.type) {
    case "load":
      await loadModel(msg);
      break;
    case "generate":
      await generate(msg.params);
      break;
    case "detect":
      runDetect(msg.params);
      break;
    case "detectText":
      detectPasted(msg.text, msg.params);
      break;
    case "interrupt":
      stoppingCriteria.interrupt();
      break;
  }
};

log.push("hello", { app: APP_VERSION, transformers: env.version });
post({ type: "hello", app: APP_VERSION, transformers: env.version, webgpuApi: !!self.navigator?.gpu });
