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
