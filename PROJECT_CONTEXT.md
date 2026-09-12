# ShadeAI C++ — Project Context (Auto-generated)

> **Останнє оновлення:** 2026-09-12
> **Git commit:** `6139af5`
> **Статус:** Chat UI + chat-dash ready, 100k діалогів сгенеровано

---

## 🎯 Мета проекту
Генеративна українська GPT (Transformer) з можливістю:
- SFT (Supervised Fine-Tuning) на діалогах
- DPO (Direct Preference Optimization)
- **Чат у браузері** з markdown, темами (vкладка "Чат" додана!)
- Дашборд з attention-хітмапами, картою слів
- Тренування на слабкому ноуті (grad accum, resume)

---

## 🏗 Архітектура моделі

| Параметр | Значення |
|----------|----------|
| **Архітектура** | Pre-LN Transformer (8 блоків) |
| **dim** | 256 |
| **blocks** | 8 |
| **ctx (max_len)** | 64 (pre-train) / 2048 (SFT) |
| **heads** | 4 (head_dim=64) |
| **FFN** | 256 → 1024 → 256 (GELU tanh-approx) |
| **Vocab** | 12,000 (BPE) |
| **Params** | ~12.5M |
| **Tokenizer** | BPE (ukr_fiction.txt, 189 MB) |

### Блок (TransformerBlock)
```
x → attn(x) → +x → ln1 → ffn → +x → ln2 → out
```
**Важливо:** attn працює на **сирому x** (без LN), ln1 — після residual. Це нестандартний pre-LN, але чекпойнти натреновані саме так.

---

## 🐛 Критичні баги (виправлені)

### 1. LayerNorm forward — row-0 bug (FIXED)
**Файл:** `src/layers/layernorm.h:36`
**Було:** `const f32* xp = x.data.data();` — для всіх рядків читало рядок 0
**Стало:** `const f32* xp = &x.data[i * dim];`
**Ефект:** вибух логітів (2900+), NaN на 1-му кроці, loss=23.0259, garbage-генерація.
**Статус:** ✅ Виправлено в `shade.exe` (build 23:19 04.09.2026)

### 2. Dashboard polling
**Файл:** `dashboard/index.html:1092`
**Було:** `setInterval(tick, 10)` — 100 req/s → `ERR_INSUFFICIENT_RESOURCES`
**Стало:** `setInterval(tick, 400)` — 2.5 req/s
**Статус:** ✅ Виправлено

---

## 📊 Чекпойнти

| Файл | Опис | best_loss | epoch |
|------|------|-----------|-------|
| `model/gpt_best.bin` | Основний (перезаписаний train26, але ваги валідні) | 5.3853 | 1 |
| `model/gpt_model.bin` | Resume checkpoint (останній збережений) | — | — |
| `model/gpt_best_run1_ep18.bin` | Старий run1 | — | 18 |
| `model/gpt_best_run2.bin` | Старий run2 | — | — |
| `model/gpt_opt.bin` | AdamW state (t=195) | — | — |

> **Нотатка:** `gpt_best.bin` метадати показують `best_loss=23.0259 epoch=1` (залишок від NaN-крашу), але **ваги коректні** — під виправленим LN дають val=5.31.

---

## 📈 Історія тренувань (ключові)

| Run | Config | Loss/Val | Статус |
|-----|--------|----------|--------|
| train14 | 21 еп., lr=0.0006, старий бінарник | 5.2–7.6 / ~5.96 | ✅ Стабільно |
| train22/23 | --continue lr=0.0002 | 4.93 / 5.33 | ✅ Зв'язний текст |
| train25/26 | NaN на ep0 (bugged LN) | 22.33 / 23.02 | ❌ NaN |
| train27 (fixed LN) | 5 еп., lr=0.0002 | 5.05→5.18 / 5.35→5.47 | ✅ Стабільно |
| train50 (running) | 50 еп., lr=0.0002, --continue | ep0: 5.11/5.54 | 🔄 В процесі |
| **chat SFT (test)** | 1 еп., lr=0.0001, 10 діалогів | 9.33 | ⚠️ Random (мало даних, баг маску) |

---

## 🔧 Поточний код (src/)

### Ключові файли
- `src/shade.cpp` — головний CLI (train gpt/chat/dpo, chat, serve, infer)
- `src/layers/transformer.h` — блоки, PE, PositionalEncoding
- `src/layers/attention.h` — SelfAttention (causal, fused softmax)
- `src/layers/layernorm.h` — **Виправлений LN** (row offset)
- `src/layers/dense.h` — Dense + Xavier init
- `src/layers/activations.h` — GELU tanh-approx
- `src/utils/embedding.h` — Embedding (scale=0.1)
- `src/utils/tokenizer_bpe.h` — BPE токенізатор
- `src/utils/serialize.h` — Бінарний формат (SHDE header)
- `src/utils/dashboard.h` — HTTP сервер + WebSocket-лайк polling

### Нові команди (додані сьогодні)
```bash
shade chat-dash --model ../model/gpt_model.bin --port 8765 --temp 0.8
# Запускає дашборд з вкладкою Чат (GPT inference через /chat)
```
```bash
shade train chat --data dialogues.jsonl --epochs 3 --ctx 2048 --lr 0.0001
shade train dpo  --data prefs.jsonl     --epochs 3 --beta 0.1
shade chat       --model gpt_model.bin --temp 0.8
shade serve      --port 8765
```

---

## 📝 Формати даних

### SFT (chat) — JSONL
```json
{"messages":[
  {"role":"system","content":"Ти — помічник українською."},
  {"role":"user","content":"Привіт!"},
  {"role":"assistant","content":"Привіт! Як допоможу?"}
]}
```
**Loss:** тільки на токенах `assistant` (mask=1).

### DPO — JSONL
```json
{"prompt":"Привіт!","chosen":"Привіт! Як справи?","rejected":"Ну шо."}
```
**Loss:** `-log(sigmoid(beta * (logprob_chosen - logprob_rejected)))`

---

## ⚙️ Конфіг тренування (train_config.json)
```json
{
  "model": { "dim": 256, "blocks": 8, "ctx": 64, "vocab": 12000 },
  "train_gpt": { "epochs": 120, "window": 10000, "lr": 0.0002, "batch_accum": 8 },
  "train_chat": { "epochs": 3, "ctx": 2048, "lr": 0.0001, "batch": 4, "grad_accum": 4 },
  "train_dpo":  { "epochs": 3, "ctx": 2048, "lr": 0.00005, "beta": 0.1, "batch": 2 },
  "laptop": { "ctx": 1024, "batch": 1, "grad_accum": 16, "fp16": false }
}
```

---

## 💻 Команди для запуску

### Pre-train (продовження)
```bash
cd build
./shade.exe train gpt --data ../data/ukr_fiction.txt --epochs 120 --window 10000 --ctx 64 --dim 256 --blocks 8 --lr 0.0002 --continue --dashboard --port 8765
```

### SFT Chat (після генерації даних)
```bash
./shade.exe train chat --data ../data/dialogues.jsonl --epochs 3 --ctx 2048 --dim 256 --blocks 8 --lr 0.0001 --batch 4 --continue --dashboard
```

### Dashboard only
```bash
./shade.exe serve --port 8765
```

### Chat CLI
```bash
./shade.exe chat --model ../model/gpt_model.bin --temp 0.8 --max-new 100
./shade.exe chat-dash --model ../model/gpt_model.bin --port 8765 --temp 0.8 --max-new 128
```

### Chat Dashboard (browser)
```bash
./shade.exe chat-dash --model ../model/gpt_model.bin --port 8765 --temp 0.8 --max-new 128
```
Откриває браузер з дашбордом і вкладкою 💬 Чат. Користувач вводить повідомлення, натискає "Надіслати", сервер запускає GPT inference і повертає відповідь.

---

## 🔄 Resume workflow (на ноуті)

1. `git clone <private-repo>`
2. `cd ShadeAI`
3. Перевірити `train_config.json` (підлаштувати `ctx`, `batch`, `grad_accum` під ноут)
4. `./resume_training.sh` — автоматично:
   - читає `train_config.json`
   - знаходить останній чекпойнт (`model/gpt_model.bin` + `gpt_opt.bin`)
   - запускає `--continue` з правильними параметрами
   - логує в `train_<дата>.log`
   - комітить зміни + лог в гіт

---

## 📋 TODO (пріоритети)

| # | Завдання | Статус |
|---|----------|--------|
| 1 | Виправити маскування в `encode_chat` (видалено gen_prompt) | ✅ Готово |
| 2 | Автогенерація діалогів з `ukr_fiction.txt` (100k+ прикладів) | ✅ Готово (100k) |
| 3 | Дашборд: вкладки Dashboard ↔ Chat, markdown, тема "шоколад" | ✅ Готово (Chat tab + CSS) |
| 4 | Чат у браузері (POST /chat, shade chat-dash) | ✅ Готово |
| 5 | SFT тренування на 100k діалогах (з 10 → 100k) | ⏳ |
| 6 | DPO trainer (повний backward через logprob) | ⏳ |
| 7 | Grad accum + int8 квантизація для ноутбука | ⏳ |
| 8 | SSE streaming для чату (замовити POST → потік відповідей) | ⏳ |

---

## 🧠 Для AI (мене) — як відновити контекст через 5 днів

1. **Прочитай цей файл** (`PROJECT_CONTEXT.md`) повністю.
2. **Перевір `git log --oneline -10`** — останні зміни.
3. **Подивись `train_config.json`** — актуальні параметри.
4. **Запусти `./resume_training.sh`** — він сам підхопить стан.
5. Якщо є нові лог-файли (`build/train_*.log`) — прочитай останні 20 рядків.
6. Продовжуй від того етапу, де зупинився (SFT → DPO → Chat UI).

---

*Файл оновлюється автоматично після кожного запуску через `auto_commit.ps1`.*