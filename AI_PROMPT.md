# AI PROMPT — Контекст для відновлення роботи (для мене, AI)

> **Цей файл читає AI (я) при старті нового чату, щоб миттєво відновити контекст проекту.**

---

## 🎯 Хто я і що роблю
Я — AI-асистент (opencode/nemotron), який допомагає розробляти **ShadeAI** — українську GPT на чистому C++ (без PyTorch/TensorFlow). Проект: Transformer з нуля, BPE токенізатор, власний дашборд, SFT/DPO тренування.

**Мова спілкування:** Українська (технічні терміни англійською).

---

## 📂 Структура проекту (must know)
```
ShadeAI C++ Test/
├── src/                    # Всі .cpp/.h (один shade.cpp + headers)
│   ├── shade.cpp           # Головний CLI (2000+ рядків)
│   ├── layers/             # transformer, attention, layernorm, dense, activations
│   ├── utils/              # tokenizer_bpe, embedding, serialize, dashboard, rag
│   └── core/               # tensor.h (AVX matmul)
├── build/                  # shade.exe, логи, .obj
├── model/                  # .bin чекпойнти (gitignored, великі)
├── data/                   # ukr_fiction.txt (189MB), bpe_vocab/merges, dialogues.jsonl
├── dashboard/              # index.html, JS, CSS (дашборд + чат)
├── PROJECT_CONTEXT.md      # <-- ГОЛОВНИЙ ФАЙЛ КОНТЕКСТУ (читай першим!)
├── train_config.json       # Всі параметри в одному місці
├── AI_PROMPT.md            # Цей файл
├── .gitignore
└── resume_training.ps1     # Авто-запуск на ноуті
```

---

## ⚡ Критично важливі факти (пам'ятай це)

### 1. LayerNorm BUG (ВИПРАВЛЕНО 04.09.2026)
- **Файл:** `src/layers/layernorm.h:36`
- **Було:** `xp = x.data.data()` — читало рядок 0 для всіх
- **Стало:** `xp = &x.data[i * dim];` + `op = out.data.data() + i * dim;`
- **Симптоми:** logits=2900, NaN на кроці 1, loss=23.0259, генерація "194194..."
- **Статус:** ✅ Фікс у поточному `shade.exe` (build 23:19 04.09)

### 2. Чекпойнти — валідні!
- `gpt_best.bin` метадати: `best_loss=23.0259 epoch=1` (залишок від NaN-крашу)
- **АЛЕ ваги коректні** — під виправленим LN дають val=5.31, loss=4.94
- `gpt_opt.bin` містить AdamW state (t=195) — resume працює

### 3. Dashboard polling
- `dashboard/index.html:1092` — `setInterval(tick, 400)` (було 10мс → 100 req/s → uBlock блокував)

### 4. Тренування зараз
- **PID 1216** — `train50` (50 еп., lr=0.0002, --continue) запущено в фону
- ep0: loss=5.11 val=5.54 → ep2: loss≈5.02 val≈5.47 (стабільно, без NaN)

---

## 🎯 Поточні завдання (пріоритет)

| # | Завдання | Статус |
|---|----------|--------|
| 1 | **Виправити `encode_chat` маскування** — loss тільки на assistant токенах | 🔄 В процесі |
| 2 | **Автогенерація діалогів** з ukr_fiction.txt (100k+ прикладів) | ⏳ |
| 3 | **Дашборд:** вкладки Dashboard ↔ Chat, markdown, шоколадна тема | ⏳ |
| 4 | **Чат у браузері** (WebSocket, markdown рендеринг) | ⏳ |
| 5 | **DPO trainer** (повний backward через logprob) | ⏳ |
| 6 | **Grad accum + config для ноутбука** | ⏳ |

---

## 🛠 Команди які я знаю

```bash
# Pre-train resume (стабільно, без NaN)
cd build && ./shade.exe train gpt --data ../data/ukr_fiction.txt --epochs 120 --window 10000 --ctx 64 --dim 256 --blocks 8 --lr 0.0002 --continue --dashboard --port 8765

# SFT Chat (після генерації diálogів)
./shade.exe train chat --data ../data/dialogues.jsonl --epochs 3 --ctx 2048 --dim 256 --blocks 8 --lr 0.0001 --batch 4 --continue --dashboard

# DPO
./shade.exe train dpo --data ../data/preferences.jsonl --epochs 3 --beta 0.1 --continue

# Dashboard only
./shade.exe serve --port 8765

# Chat CLI
./shade.exe chat --model ../model/gpt_model.bin --temp 0.8
```

---

## 📋 Як я працюю (workflow)

1. **Читаю `PROJECT_CONTEXT.md`** — головний стан
2. **Дивлюсь `git log --oneline -5`** — що змінилося
3. **Читаю `train_config.json`** — параметри
4. **Перевіряю `build/train_*.log`** — останні епокхи
5. **Запускаю `./resume_training.ps1`** або відповідну команду
6. **Після кожного запуску** — оновлюю `PROJECT_CONTEXT.md`, комітлю

---

## 💡 Мої принципи
- **Коротко, по суті** — без води
- **Код > коментарі** — не пишу зайвих коментарів у код
- **Виправляю баги, а не обходжу** — root cause
- **Автоматизація** — все що повторюється > скрипт/конфіг
- **Приватність** — гіт локальний/приватний, чекпойнти не в гіт

---

## 🚨 Якщо щось пішло не так
- **NaN знову?** → перевір `layernorm.h:36` (row offset)
- **Dashboard не грузиться?** → uBlock exception `localhost:8765/*` + polling 400ms
- **Loss не падає?** → перевір маскування в `encode_chat` / дані
- **Чекпойнт не завантажується?** → перевір `serialize.h` header (SHDE, dim, blocks, ctx, count)

---

## 📝 Нотатки для себе (оновлюй цей файл)

- [ ] Виправити `encode_chat` — правильне маскування assistant токенів
- [ ] Створити `generate_dialogues.cpp` → 100k+ прикладів з ukr_fiction
- [ ] Дашборд: tabs (Dashboard/Chat), markdown (marked.js), theme toggle
- [ ] Чат: WebSocket, streaming, markdown рендеринг
- [ ] DPO: правильний backward через logprob ratio
- [ ] Grad accum в `AdamW` + `laptop` конфіг в train_config.json

---

*Останнє оновлення: 2026-09-06 23:45*
*Наступний сеанс: прочитай це, потім `git log --oneline -5`, потім `cat PROJECT_CONTEXT.md`*