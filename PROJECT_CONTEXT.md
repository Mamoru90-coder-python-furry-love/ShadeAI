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
tests/gradcheck.cpp        — central finite differences для Dense/LayerNorm/GRU/Attention (включно з h0), Embedding, TransformerBlock, маленького GPT (2 блоки), CharRNN (embedding→GRU BPTT→dense) + cross-entropy dlogits; NaN-guard self-test
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
Dense           OK  worst rel err 1.18e-05
LayerNorm       OK  worst rel err 2.42e-05
GRU             OK  worst rel err 1.80e-05
Attention       OK  worst rel err 2.55e-05
Embedding       OK  worst rel err 9.90e-06
TransformerBlock OK worst rel err 2.68e-05
GPT             OK  worst rel err 3.48e-05
GPTLoss         OK  worst rel err 1.72e-04
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

Усі цілі числа фіксованої ширини `int32_t`/`uint32_t`, порядок байтів little-endian (x86/x64); ваги — IEEE-754 binary32. **Фактичний magic у файлі — `47 44 48 53` = `GDHS`** (ціле `0x53484447` у little-endian), а не буквальний ASCII `SHDG`. Це legacy wire signature v1/v2; значення збережено для сумісності, міняти його без нової версії формату заборонено.

**SHDG v1 (лише читання):** `magic(4)='SHDG' | version(4)=1 | vocab(4) | dim(4) | blocks(4) | ctx(4) | tokenizer[i32 n; n×(i32 len; len bytes)] | i32 tensor_count | tensors[i32 rows, i32 cols, rows*cols floats]`. Заголовок 24 байти, checksum відсутній. Фактичну структуру зафіксовано стабільним бінарним fixture `tests/fixtures/shdg_v1_min.bin` (1408 байт: vocab=8, dim=4, blocks=1, ctx=4, 15 тензорів, вага `i = 0.01f*(i%5)`); генератор — `tests/fixtures/gen_v1_fixture.cpp`.

**SHDG v2 (запис і читання):** та сама структура, але `u32 crc32` вставляється на offset 8 і config зсувається на offset 12. CRC-32 (IEEE 802.3, відбитий поліном `0xEDB88320`, init/final `0xFFFFFFFF`) покриває **всі байти файлу, крім самого поля checksum** — тобто конфіг, tokenizer metadata й усі ваги. Обчислюється потоково буфером 64 KiB (`gpt_compute_file_crc`), без алокацій залежно від розміру файлу; той самий хелпер використовується і при записі (tmp-файл відкривається `w+b`, CRC дописується перед `atomic_replace`).

**Правила сумісності:** v1 — завантажується без вимоги checksum, ніколи не перезаписується у v1; v2 — завантажується з перевіркою CRC, нові збереження тільки у v2; невідомі версії відхиляються. Автоматичної конвертації ваг немає.

**Бюджет tokenizer metadata:** `Tokenizer::read(f, max_bytes)` (за замовчуванням 64 MiB) списує витрати до кожної алокації: `2*sizeof(std::string) + 48` на слово (об'єкти `id2word`/`word2id`, вузол і бакет map) плюс `2*(len+1)` за дані й дубльований ключ. Перед `reserve(n)` перевіряється мінімальна вартість, перед кожним `len` — що файл реально містить стільки байтів (`ftell`/залишок); додатково `len ≤ 1 MiB`, `n ≤ 10⁶`. `read_gpt_checkpoint(..., max_tokenizer_bytes)` передає свій ліміт. Duplicated токени відхиляються (`map.emplace().second`), спеціальні ID 0..3 перевіряються. Стан `Tokenizer` не змінюється до повного успіху.

**Важливе уточнення:** 64 MiB — це **логічний бюджет за моделлю оцінювання**, а не гарантована верхня межа фактично виділеної RAM. `NODE_AND_BUCKET = 48` — емпірична оцінка накладних витрат `unordered_map`; реальний розмір вузла, bucket-масиву й алокацій відрізняється між MSVC STL, libstdc++ і libc++. Ліміт надійно відсікає патологічні випадки (1e6 слів × 1 MiB, оголошена довжина без даних у файлі), але не є точним обмеженням споживання пам'яті на всіх реалізаціях STL. Для суворого контролю знадобиться власний компактний формат словника замість `unordered_map` (окрема задача, не частина P1).

**Порядок перевірок завантаження:** magic → версія → (v2: CRC до будь-якого парсингу) → межі конфігу → `gpt_config_budget()` → tokenizer (з бюджетом) → кількість/форми тензорів → EOF → **commit**. Усі дані стейджаться локально й присвоюються викликачу лише на повному успіху; checksum **не замінює** валідацію розмірів/форм/NaN/бюджету. Трейлінгові байти відхиляються.

**Обмеження CRC:** виявляє випадкове пошкодження, не захищає від навмисної підміни (не криптографічний хеш). Інші формати (`Serialize::save_model`, optimizer state) checksum поки не мають.

**Тести:** `tests/checkpoint_integrity_check.cpp` — валідний v2 round-trip + детермінізм байтів, pin на wire-байти `GDHS`, v1 fixture з контрольними вагами та ре-сейв у v2, нульова/коротка/`bad magic`/невідома версія/обрізані/трейлінгові файли, бітий checksum, перевернутий байт ваги; окрема група з **перерахованим CRC** (`absurd vocab`, `INT_MAX dim`, combined budget, зіпсований `<PAD>`, неправильний `tensor_count`, oversized/negative dims, NaN-вага, `len=1MiB` без даних у файлі) доводить, що валідація працює незалежно від checksum. Обмежений бюджет tokenizer на валідному файлі, дублікати токенів, відмова фінальної заміни (не залишає `.tmp`) і побайтова незмінність попереднього чекпойнту після невдалого запису. Кожен негативний випадок перевіряє, що стан викликача не змінився.

#### Gradient checking — P2

`tests/gradcheck.cpp` (складається з тим самим прапорцем, що й у `build\test.bat`): скінченна різниця `eps=1e-3`, відносна похибка з знаменником `max(1,|num|,|ana|)`, поріг `GRAD_TOL=2e-3`. Покрито: Dense, LayerNorm, GRU, SelfAttention (causal), **Embedding** (лише `w`/`gw` — входи дискретні, `backward()` повертає порожній градієнт), **TransformerBlock** (усі 12 параметрів + градієнт по входу `x`, включно з residual-потоком і GELU), **маленький GPT** (8 vocab / dim 4 / 2 блоки / ctx 8 — 27 груп параметрів; 2 блоки спеціально, щоб перевірити зворотний обхід `GPT::backward`). Окремо `GPTLoss` звіряє `dlogits` із `GPT::compute_loss` (cross-entropy, шлях реального тренування) проти скінченної різниці скалярного mean-loss.

`gpt.cpp` включається в тест з `#define main gpt_cli_main`, тобто перевіряється **реальна** композиція моделі, а не її копія. Градієнти, що накопичуються (`Dense/Attention/Embedding` через `+=`), коректно обнуляються харнесом перед єдиним аналітичним `backward`; числові `forward()` під час sweep не пишуть у градієнти. Тренувальний цикл (`opt.zero_grad` на початку епохи й після `opt.step`) перевірено окремо — накопичення між мікро-батчами є навмисним і далі ділиться на `iters_since_step`.

Embedding використовує послідовність із **повторюваними ID** (`0,1,0,2,1,0,3,1,4,5,6,7`), щоб перевірити накопичення градієнта в `gw` для одного токена; усі рядки vocab присутні.

**NaN-guard.** Порівняння `NaN > worst` повертає `false`, тож без явної перевірки нечисловий градієнт міг би непомітно лишити `worst = 0` і дати хибний `OK`. Харнес (`gradcheck` і `gpt_loss_gradcheck`) тепер має `finite_ok` і перевіряє `std::isfinite(num)`, `std::isfinite(a)` та `std::isfinite(err)`; будь-яка нечислова величина друкує діагностику й робить результат `FAIL`. Регресійний self-test (`nan_guard_selftests`) підсовує NaN в аналітичний і в числовий градієнт окремо — обидва мають бути відхилені. Sentinel вимагає саме причини відмови: `expect_fail` повертає `!ok && nonfinite_seen`, тож відмова з будь-якої іншої причини не зараховується. Емпірично перевірено двічі: (1) з тимчасово вимкненою перевіркою `isfinite` обидва sentinel-и друкують `not rejected` і тест падає з `exit=1`; (2) якщо прибрати лише встановлення `nonfinite_seen` (залишивши `finite_ok = false`), sentinel-и знову падають — тобто вони справді перевіряють причину, а не просто факт відмови. З повним кодом обидва проходять тихо.

Фактичні найгірші відносні похибки: Dense 1.18e-05, LayerNorm 2.42e-05, GRU 1.80e-05, Attention 2.55e-05, Embedding 9.90e-06, TransformerBlock 2.68e-05, GPT 3.48e-05, GPTLoss 1.72e-04 — усі нижче порогу.

#### Gradient checking char-RNN — P2 (продовження)

`tests/gradcheck.cpp` підключає `src/demos/char_rnn.cpp` напряму (у демо немає `main`), тому перевіряється **реальна** композиція `CharRNN::forward/backward`: `Embedding → GRU (BPTT по всій послідовності) → Dense`. Конфіг 8 vocab / embed 4 / hidden 6 / seq 5, 12 груп параметрів (`embed.w`, `fc.w`, `fc.b`, 9 GRU-матриць/ biases), повторювані ID у послідовності. Входи дискретні → лише параметри. Найгірша відносна похибка 3.43e-05. Разом із GPT це вже дві архітектури ShadeAI з перевіреними градієнтами.

**Спостереження (не виправлялось у цьому коміті):** у навчальному циклі `demo_charrnn()` (рядки з обчисленням `dlogits`) градієнт cross-entropy рахується як `p - onehot` **без** ділення на `seq_len`, тоді як `total_loss` ділиться на `seq_len`. Тобто оптимізується сума CE за послідовність, а друкується середня — на відміну від `GPT::compute_loss`, де масштабування `1/T` є в обох. Практично це згладжується `clip_grads(1.0f)` + AdamW, але це розузгодження варто виправити окремо, якщо потрібна точна відповідність друкованого loss до цілі оптимізації.

### Ще не зроблено

- Gradient checking покриває GPT і char-RNN, але не покриває інші demo-моделі (MNIST MLP/CNN, transformer_lm/transformer_test).
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
4. Прогнати: `build\test.bat` (simd_check, gradcheck, serialize_checks, gpt_checkpoint_check, checkpoint_integrity_check, memtrack_check, tensor_checks, demos).
5. Не додавати нові архітектури, доки gradient checking не покриє решту demo-композицій (char-RNN/GRU) — GPT уже покрито.

---

## 🗄 Історичний контекст (не відповідає поточному дереву)

Стара версія цього файлу описувала `src/shade.cpp`, чат у браузері, SFT/DPO, 8-блоковий GPT і checkpoint-файли з `model/*.bin`. Ці файли/команди були видалені або не зібрані в поточному `main`. Історія збережена в git, але орієнтуватися на неї як на активний стан не варто.
