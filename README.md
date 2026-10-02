# LLM Watermarking Playground

Локальная переработка [nohypeai/watermarking-playground](https://huggingface.co/spaces/nohypeai/watermarking-playground).

Инференс идёт в браузере: Transformers.js и ONNX Runtime. Водяной знак ставится на логиты во время генерации. Сейчас это Gemma 4 E4B и турнирное семплирование в духе SynthID-Text, плюс зелёный/красный список Kirchenbauer.

Оригинал — статический Space на Hugging Face. Модели там не лежали в репозитории страницы: браузер качал их с Hugging Face Hub. Здесь то же самое: если весов нет рядом со страницей, они скачиваются с `huggingface.co` при первом открытии.

## Запуск

Дважды щёлкните `start.bat`. Откроется http://127.0.0.1:8765/

Закрытие окна останавливает сервер. Не открывайте `index.html` двойным щелчком: воркеры и WebGPU с `file://` не работают. Заголовки изоляции выставляет `serve.py`.

## Онлайн

GitHub Pages: Settings → Pages → Branch `main` → `/ (root)`. Адрес будет `https://belerafon.github.io/LLM_watermark_payground/`.

Посетитель ничего не качает руками. Браузер сам берёт `onnx-community/gemma-4-E4B-it-ONNX` с Hugging Face. Это около 5 ГБ в кэш браузера, один раз. Нужны Chrome или Edge с WebGPU.

GitHub Pages не ставит заголовки изоляции для WebGPU. Их добавляет `coi-serviceworker.js` (страница один раз перезагрузится).

Локально по-прежнему `start.bat`. Если веса уже лежат в `models/`, страница возьмёт их и в сеть не пойдёт. Докачка: `download_gemma4.ps1`.

## Что в репозитории

- `index.html`, `main.js`, `worker-v4.js`, `watermark.js`, `models.js`, `style.css` — страница
- `vendor4/` — Transformers.js 4.3.0 и WASM ONNX Runtime, чтобы страница не ходила в CDN
- `serve.py`, `start.bat`, `start.ps1` — локальный сервер
- `download_gemma4.ps1` — докачка весов
