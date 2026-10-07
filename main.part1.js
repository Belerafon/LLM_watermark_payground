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
