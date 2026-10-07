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
