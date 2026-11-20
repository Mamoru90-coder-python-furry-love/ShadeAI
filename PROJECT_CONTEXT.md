# ShadeAI C++ — Project Context

> **Оновлено:** 2026-11-20  
> **Поточний коміт:** див. `git log -1 --oneline`  
> **Статус:** локальний neural-net рушій з нуля; CLI збирається з `src/gpt.cpp`.

---

## 🎯 Мета

Локальна необмежена нейромережа, написана з нуля на C++ без готових архітектур чи моделей.

- C++ — архітектура, autograd/backprop, тренування.
- AVX1/OpenMP — окремі обчислювальні ядра (не весь код).
- Ручний ASM — можливий пізніше тільки для hottest Kernel-ів.
- Розуміння крипторинку — окремий майбутній етап, зараз у коді відсутнє.

---

## 🧱 Що реально є в дереві

### Збірка

`build/compile.bat` шукає MSVC через `vswhere`, передає `vcvars64.bat` і збирає:

```
src/gpt.cpp  ->  build/shade.exe
```

Прапорці: `/std:c++17 /O2 /arch:AVX /openmp /EHsc`.

`src/main.cpp` — окрема демо-програма (XOR, self-attention copy task, save/load).
`src/main_rnn.cpp` та `src/demos/*` — окремі demos, не входять у `shade.exe`.

### CLI `gpt.cpp`

Актуальний CLI:

```
build\shade.exe                                        # train: data/tinystories_valid.txt -> gpt_model.bin
build\shade.exe --train [data.txt] [model.bin]          # train з явними шляхами
build\shade.exe --load <model.bin> "prompt"             # inference
build\shade.exe --load <model.bin> --prompt "prompt"    # inference (еквівалентно)
```

Checkpoint тепер **самостійний**: magic/version + конфіг (`vocab/dim/blocks/ctx`) + tokenizer + ваги.
Тому `--load` відбудовує архітектуру й словник без навчального тексту. Старі формати (лише ваги) відхиляються з повідомленням.

Поточний навчальний конфіг у коді: `dim=64`, `blocks=3`, `max_len=32`, `vocab` будується з тексту (до 2000 токенів). `train_config.json` не читається.

### Ядро

| Файл | Призначення |
|------|-------------|
| `src/core/tensor.h` | `Tensor`, broadcasting, matmul, runtime shape/bounds checks |
| `src/core/simd.h` | AVX1 kernel-и: dot, exp, softmax, scalar fallback |
| `src/core/loss.h` | vectorized cross-entropy з averaging на batch |
| `src/optimizer/adamw.h` | виправлений AdamW з decoupled weight decay |
| `src/layers/attention.h` | causal self-attention, SIMD QKᵀ/softmax/score·V |
| `src/layers/transformer.h` | TransformerBlock, виправлений sinusoidal positional encoding |
| `src/layers/layernorm.h`, `dense.h`, `gru.h` | шари з backward |
| `src/utils/serialize.h` | SHDE checkpoint, транзакційне читання, перевірка `fread`/NaN/Inf |
| `src/utils/tokenizer.h` | словниковий tokenizer, бінарне збереження у checkpoint |
| `src/utils/tokenizer_bpe.h` | окремий BPE tokenizer (не використовується в `gpt.cpp`) |

---

## ✅ Що перевірено

### Тести

```
tests/simd_check.cpp       — AVX vs scalar: exp/dot/softmax/matmul/attention/loss/AdamW
tests/gradcheck.cpp        — central finite differences для Dense/LayerNorm/GRU/Attention, включно з h0
tests/serialize_checks.cpp — обрізані/пошкоджені/з зайвими байтами checkpoint-и
tests/gpt_checkpoint_check.cpp — round-trip, top-k (включно з однаковими ймовірностями), бюджет конфігу, атомарний запис, пошкоджені файли
tests/tensor_checks.cpp    — runtime перевірки навіть при /DNDEBUG
tests/bench.cpp            — scalar vs AVX: matmul/attention/AdamW
```

Зручна перевірка всього:

```
build\test.bat
```

Останній результат `gradcheck` ( tolerance `2e-3` ):

```
Dense        OK  worst rel err 1.18e-05
LayerNorm    OK  worst rel err 2.42e-05
GRU          OK  worst rel err 1.80e-05
Attention    OK  worst rel err 2.55e-05
```

### Виправлені математичні помилки

- AdamW: `km = lr*ib1`, `kw = lr*wd`; decay без bias correction.
- Positional encoding: частота залежить від пари координат `(j & ~1) / dim`.
- Cross-entropy: `dlogits` ділиться на batch; у `gpt.cpp` — ще на `T`.
- Градієнти в `gpt.cpp` усереднюються перед `clip_grads`/`step`.
- GRU: `gUn` використовує `r_j`, а не `r_k`.
- Attention backward: `dA = dO @ Vᵀ`, `dV = Aᵀ @ dO`.
- LayerNorm backward: коректний `dx` з урахуванням `gamma`.
- Tensor: neg dimensions, overflow `rows*cols`, bounds, matmul/broadcast mismatch — кидають винятки.
- Serialization: транзакційне завантаження — спершу читання в тимчасові буфери, потім перевірка всього файлу (включно з EOF) і тільки тоді заміна ваг. Відхиляються обрізані, NaN/отруєні, з зайвими байтами, з неправильною кількістю/формою параметрів файли. Те саме для optimizer state.
- `Tensor::slice()`: перевірка `start + count` без переповнення.
- Перевірки параметрів конструкторів і форм входу в Dense/LayerNorm/GRU/Attention/Transformer.
- GPT checkpoint: конфіг + tokenizer + ваги в одному файлі; `--load` відбудовує модель і генерує текст. `sample()` більше не має фіксованого буфера на 16384 токени.
- Top-k: лишається **рівно** `top_k` токенів; однакові ймовірності ламаються за індексом (раніше могло лишитись `top_k + 1`).
- `Tokenizer::read()` перевіряє, що ID 0..3 — це `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`.
- Бюджет checkpointу обчислюється з конфігу **до** `make_unique<GPT>()`: точний перелік форм 27 тензорів (3 + 12 на блок), кількість параметрів, ваги + градієнти + кеші транспонувань Dense. Уся арифметика — через `ck_mul`/`ck_add` з перевіркою переповнення `int64` і відхиленням від'ємних операндів.
- **Memory ledger (P0, виведено з коду, не на око).** Постійні activation-кеші одного блоку `P = 28*T*dim + 6*T² + 4*T` (Attention: 6 Td + 6 T²; 2×LayerNorm: 2 Td + 4 T; Dense ff1/ff2: 10 Td; GELU: 4 Td; поля блоку: 6 Td). Тимчасові піки **одного** блоку: forward `F = max(T² + Td, 10*Td)` (вікна attention-head і FFN не перетинаються); backward `W = 21*Td + 2*T²` (16 Td зовнішніх локальних, що вже існують на піку вкладених `attn.backward`, + 5 Td Attention + dA/dScores). **Глобальний пік:** `B*P + max(F, W) + G`, де `G = 4*T*vocab + 3*T*dim` (output Dense, logits/dlogits, pe/h). Ключова поправка: тимчасові буфери різних блоків **не живуть одночасно** (кожен виклик блоку повертається до старту наступного), тому transient враховується **один раз**, а не множиться на `B` — попередня формула `B*(P+max(F,W))+G` була надмірно консервативною.
- Окремі бюджети (ліміти — **кількість float-елементів**, не байти): побудова моделі `params*2 + caches ≤ 50M` (200 MB); завантаження checkpointу `+ params` (staged ваги співіснують із новою моделлю) `≤ 75M` (300 MB); активації `≤ 20M` (80 MB). `read_tensor()` перевіряє форму й **залишок** бюджету до виділення буфера.
- Тести ledger-у: (1) формула — кожен phase-пік окремо (`block_persistent_act`, `block_forward_transient`, `block_backward_transient`), **`B*P + max(F,W) + G`**, масштабування лише persistent з 1/2/8/100 блоків, великий ctx/dim/vocab, точна межа, переповнення `ck_mul`/`ck_add`; (2) `tests/memtrack_check.cpp` — debug-only лічильник живих `Tensor`-буферів (`SHADE_MEM_TRACK`, тільки в тестовому TU) вимірює **реальний** пік forward+backward і доводить, що він росте рівно на `P` на блок, а не на `P+transient`. Тест перевіряє, що виведені з конфігу форми й лік параметрів збігаються з живою `GPT::params()` (6927 параметрів / 27 тензорів).
- Атомарний запис GPT checkpointу: `*.tmp` + `MoveFileExA(MOVEFILE_REPLACE_EXISTING)` на Windows; попередній файл не пошкоджується, якщо запис перервався.

#### Формат GPT checkpointу (SHDG) — P1

Усі цілі числа фіксованої ширини `int32_t`/`uint32_t`, порядок байтів little-endian (x86/x64); ваги — IEEE-754 binary32.

**SHDG v1 (лише читання):** `magic(4)='SHDG' | version(4)=1 | vocab(4) | dim(4) | blocks(4) | ctx(4) | tokenizer[i32 n; n×(i32 len; len bytes)] | i32 tensor_count | tensors[i32 rows, i32 cols, rows*cols floats]`. Заголовок 24 байти, checksum відсутній. Фактичну структуру зафіксовано стабільним бінарним fixture `tests/fixtures/shdg_v1_min.bin` (1408 байт: vocab=8, dim=4, blocks=1, ctx=4, 15 тензорів, вага `i = 0.01f*(i%5)`); генератор — `tests/fixtures/gen_v1_fixture.cpp`.

**SHDG v2 (запис і читання):** та сама структура, але `u32 crc32` вставляється на offset 8 і config зсувається на offset 12. CRC-32 (IEEE 802.3, відбитий поліном `0xEDB88320`, init/final `0xFFFFFFFF`) покриває **всі байти файлу, крім самого поля checksum** — тобто конфіг, tokenizer metadata й усі ваги. Обчислюється потоково буфером 64 KiB (`gpt_compute_file_crc`), без алокацій залежно від розміру файлу; той самий хелпер використовується і при записі (tmp-файл відкривається `w+b`, CRC дописується перед `atomic_replace`).

**Правила сумісності:** v1 — завантажується без вимоги checksum, ніколи не перезаписується у v1; v2 — завантажується з перевіркою CRC, нові збереження тільки у v2; невідомі версії відхиляються. Автоматичної конвертації ваг немає.

**Порядок перевірок завантаження:** magic → версія → (v2: CRC до будь-якого парсингу) → межі конфігу → `gpt_config_budget()` → tokenizer → кількість/форми тензорів → EOF → **commit**. Усі дані стейджаться локально й присвоюються викликачу лише на повному успіху; checksum **не замінює** валідацію розмірів/форм/NaN/бюджету. Трейлінгові байти відхиляються.

**Обмеження CRC:** виявляє випадкове пошкодження, не захищає від навмисної підміни (не криптографічний хеш). Інші формати (`Serialize::save_model`, optimizer state) checksum поки не мають.

**Тести:** `tests/checkpoint_integrity_check.cpp` — валідний v2 round-trip + детермінізм байтів, v1 fixture з контрольними вагами та ре-сейв у v2, нульова/коротка/`bad magic`/невідома версія/обрізані/трейлінгові файли, бітий checksum, перевернутий байт ваги; окрема група з **перерахованим CRC** (`absurd vocab`, `INT_MAX dim`, combined budget, зіпсований `<PAD>`, неправильний `tensor_count`, oversized/negative dims, NaN-вага) доводить, що валідація працює незалежно від checksum. Кожен негативний випадок перевіряє, що стан викликача не змінився.

### Ще не зроблено

- Gradient checking не покриває весь TransformerBlock/Embedding/GRU в складі GPT.
- `train_config.json` не читається `gpt.cpp` — конфіг зашитий у коді.
- Інші шляхи запису (`Serialize::save_model`, `AsyncSaver`) ще не атомарні; GPT checkpoint — уже так.

---

## 📊 Поточний стан даних/чекпойнтів

- У `model/` лежать лише невеликі `.txt` з метаданими loss; `.bin`-чекпойнти у `.gitignore` і в дереві відсутні.
- `train_config.json` описує 8 блоків, dim=256, ctx=64, vocab=12000, але цей конфіг **не читається** поточним `gpt.cpp`.
- Відсутні файли, на які посилається старий пайплайн: `data/ukr_fiction.txt`, `data/dialogues.jsonl`, `data/bpe_vocab.txt`, `data/bpe_merges.txt`.
- Потрібна регенерація даних/BPE або оновлення `gpt.cpp`, перш ніж запускати велике тренування.

---

## 🧭 Для AI/мене — відновлення контексту

1. `git log -5 --oneline`.
2. Прочитати цей файл і `PLAN.md`.
3. Зібрати: `build\compile.bat`.
4. Прогнати: `build\test.bat` (simd_check, gradcheck, serialize_checks, gpt_checkpoint_check, tensor_checks, demos).
5. Не додавати нові архітектури, доки gradient checking не покриє решту шарів і TransformerBlock повністю.

---

## 🗄 Історичний контекст (не відповідає поточному дереву)

Стара версія цього файлу описувала `src/shade.cpp`, чат у браузері, SFT/DPO, 8-блоковий GPT і checkpoint-файли з `model/*.bin`. Ці файли/команди були видалені або не зібрані в поточному `main`. Історія збережена в git, але орієнтуватися на неї як на активний стан не варто.
