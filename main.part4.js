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
