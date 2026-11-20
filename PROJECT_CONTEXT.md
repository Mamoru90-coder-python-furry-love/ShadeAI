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
tests/gradcheck.cpp        — central finite differences для Dense/LayerNorm/GRU/Attention (включно з h0), Embedding, TransformerBlock, маленького GPT (2 блоки), CharRNN (embedding→GRU BPTT→dense), MNIST MLP (Dense→ReLU→Dense→ReLU→Dense), MNIST CNN (Conv→ReLU→MaxPool ×2→Dense→Dense), TransformerLM (embedding→pos-encoding→2 блоки→dense) + cross-entropy dlogits (GPTLoss, CharRNNLoss, TransformerLMLoss) + окремий CrossEntropyLoss::forward (AVX-гілка, різні n/C, double-еталон, великі logits; CE) + валідація входів CharRNN::compute_loss / TransformerLM::compute_loss / CrossEntropyLoss::forward (CharRNNVal, TransformerLMVal, CEVal); NaN-guard self-test
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
CharRNN         OK  worst rel err 3.43e-05
MLP             OK  worst rel err 9.11e-06
CNN             OK  worst rel err 3.47e-06
TransformerLM   OK  worst rel err 1.47e-05
GPTLoss         OK  worst rel err 1.60e-04
CharRNNLoss     OK  worst rel err 1.44e-04
TransformerLMLoss OK worst rel err 1.22e-04
CharRNNVal       OK
TransformerLMVal OK
CE n1 C8         OK  ref 7.29e-08  fd 1.44e-04
CE n3 C5         OK  ref 6.35e-08  fd 9.05e-05
CE n4 C9         OK  ref 6.23e-08  fd 1.15e-04
CE n8 C16        OK  ref 9.50e-08  fd 1.91e-04
CE n2 C7         OK  ref 6.11e-08  fd 6.78e-05
CE bigbase       OK  ref 2.98e-08 (ref only)
CE wide          OK  ref 9.78e-09 (ref only)
CEVal            OK
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

`tests/gradcheck.cpp` (складається з тим самим прапорцем, що й у `build\test.bat`): скінченна різниця `eps=1e-3`, відносна похибка з знаменником `max(1,|num|,|ana|)`, поріг `GRAD_TOL=2e-3`. Покрито: Dense, LayerNorm, GRU, SelfAttention (causal), **Embedding** (лише `w`/`gw` — входи дискретні, `backward()` повертає порожній градієнт), **TransformerBlock** (усі 12 параметрів + градієнт по входу `x`, включно з residual-потоком і GELU), **маленький GPT** (8 vocab / dim 4 / 2 блоки / ctx 8 — 27 груп параметрів; 2 блоки спеціально, щоб перевірити зворотний обхід `GPT::backward`). Окремо `GPTLoss` і `CharRNNLoss` звіряють `dlogits` із `GPT::compute_loss` / `CharRNN::compute_loss` (cross-entropy, шлях реального тренування) проти скінченної різниці скалярного mean-loss. Обидва loss-харнеси мають тотожний NaN-guard, що й загальний `gradcheck`. Також перевірено **MNIST MLP** — реальну композицію `Dense → ReLU → Dense → ReLU → Dense` з демо `mnist_mlp.cpp` (6 груп параметрів + градієнт по входу), **MNIST CNN** — реальну композицію `Conv → ReLU → MaxPool ×2 → Dense → Dense` з демо `mnist_cnn.cpp` (8 груп параметрів + градієнт по входу), та **TransformerLM** — реальну композицію `Embedding → PositionalEncoding → TransformerBlock ×2 → Dense` з демо `transformer_lm.cpp` (15 груп параметрів; входи дискретні). `TransformerLMLoss` окремо звіряє `dlogits` із новим `TransformerLM::compute_loss`, а `TransformerLMVal` перевіряє ті самі 10 видів некоректного вводу, що й `CharRNNVal` (детерміновані дані, без споживання RNG). Окремо додано `CrossEntropyLoss::forward` — див. розділ нижче. Демо `transformer_test.cpp` не додає нової композиції — там лише `TransformerBlock → Dense`, уже покрите окремо.


`gpt.cpp` включається в тест з `#define main gpt_cli_main`, тобто перевіряється **реальна** композиція моделі, а не її копія. Градієнти, що накопичуються (`Dense/Attention/Embedding` через `+=`), коректно обнуляються харнесом перед єдиним аналітичним `backward`; числові `forward()` під час sweep не пишуть у градієнти. Тренувальний цикл (`opt.zero_grad` на початку епохи й після `opt.step`) перевірено окремо — накопичення між мікро-батчами є навмисним і далі ділиться на `iters_since_step`.

Embedding використовує послідовність із **повторюваними ID** (`0,1,0,2,1,0,3,1,4,5,6,7`), щоб перевірити накопичення градієнта в `gw` для одного токена; усі рядки vocab присутні.

**NaN-guard.** Порівняння `NaN > worst` повертає `false`, тож без явної перевірки нечисловий градієнт міг би непомітно лишити `worst = 0` і дати хибний `OK`. Харнеси (`gradcheck`, `gpt_loss_gradcheck`, `charrnn_loss_gradcheck`) тепер мають `finite_ok` і перевіряють `std::isfinite(num)`, `std::isfinite(a)` та `std::isfinite(err)`; будь-яка нечислова величина друкує діагностику й робить результат `FAIL`. Регресійний self-test (`nan_guard_selftests`) підсовує NaN в аналітичний і в числовий градієнт окремо — обидва мають бути відхилені. Sentinel вимагає саме причини відмови: `expect_fail` повертає `!ok && nonfinite_seen`, тож відмова з будь-якої іншої причини не зараховується. Емпірично перевірено двічі: (1) з тимчасово вимкненою перевіркою `isfinite` обидва sentinel-и друкують `not rejected` і тест падає з `exit=1`; (2) якщо прибрати лише встановлення `nonfinite_seen` (залишивши `finite_ok = false`), sentinel-и знову падають — тобто вони справді перевіряють причину, а не просто факт відмови. З повним кодом обидва проходять тихо.

Фактичні найгірші відносні похибки: Dense 1.18e-05, LayerNorm 2.42e-05, GRU 1.80e-05, Attention 2.55e-05, Embedding 9.90e-06, TransformerBlock 2.68e-05, GPT 3.48e-05, CharRNN 3.43e-05, MLP 9.11e-06, CNN 3.47e-06, TransformerLM 1.47e-05, GPTLoss 1.60e-04, CharRNNLoss 1.44e-04, TransformerLMLoss 1.22e-04 — усі нижче порогу.

#### Gradient checking char-RNN — P2 (продовження)

`tests/gradcheck.cpp` підключає `src/demos/char_rnn.cpp` напряму (у демо немає `main`), тому перевіряється **реальна** композиція `CharRNN::forward/backward`: `Embedding → GRU (BPTT по всій послідовності) → Dense`. Конфіг 8 vocab / embed 4 / hidden 6 / seq 5, 12 груп параметрів (`embed.w`, `fc.w`, `fc.b`, 9 GRU-матриць/ biases), повторювані ID у послідовності. Входи дискретні → лише параметри. Найгірша відносна похибка 3.43e-05. Разом із GPT це вже дві архітектури ShadeAI з перевіреними градієнтами.

#### Нормалізація loss/градієнта в CharRNN (виправлено)

Раніше в навчальному циклі `demo_charrnn()` градієнт cross-entropy був `p - onehot` **без** `/seq_len` (сума CE), а друкуваний `total_loss` ділився на `seq_len` (середнє) — оптимізувалась одна ціль, друкувалась інша. Це впливало на `clip_grads(1.0f)`, бо масштаб суми відрізнявся від масштабу середнього. Виправлено окремим комітом:

- додано `CharRNN::compute_loss(logits, targets, dlogits)` за аналогією з `GPT::compute_loss`: повертає **середню** CE по токенах і пише `dlogits = (p - onehot)/T`, тобто градієнт саме повернутого mean-loss;
- навчальний цикл тепер викликає `compute_loss` замість інлайн-обчислення, а накопичені по вікнах градієнти ділить на `iters` **до** `clip_grads` і `opt.step` — так optimizer бачить градієнт того самого mean-loss, що друкується (`total_loss / iters`);
- `CharRNNLoss` окремо звіряє `dlogits` із finite-difference скалярного mean-loss (найгірша похибка 1.44e-04).

**Валідація входів `CharRNN::compute_loss`.** Оскільки функція тепер доступна поза демо, додано перевірки на початку: `vocab_size > 0`, `logits.cols == vocab_size`, `logits.rows > 0`, `targets.size() == logits.rows`, і кожен target — скінченне ціле в межах `[0, vocab_size)`. Без них некоректні дані давали б вихід за межі рядка або ділення на нуль.

Критичний порядок: **спочатку перевірка сирого `f32`, і лише потім `(int)`**. Приведення `float → int` має невизначену поведінку для NaN, ±Inf і значень поза діапазоном `int`, тож попередня версія (_cast перед перевіркою_) була небезпечною попри наявність guard-а. Перевірка uses `std::isfinite(raw) && raw >= 0 && raw < vocab_size && std::trunc(raw) == raw`, після чого cast у основному циклі безпечний.

`CharRNNVal` перевіряє, що всі 10 видів некоректного вводу відхиляються через `std::invalid_argument`: неправильна ширина logits, порожня послідовність, розсинхрон довжин, від'ємний ID, ID ≥ vocab, дробовий ID, NaN, +Inf, −Inf і велике скінченне значення (1e30). Нечислові edge-кейси перевіряють саме небезпечний порядок cast-у. Guard `vocab_size > 0` лишився як захисне, але **не тестується напряму**: `Dense` у голові моделі ще на конструкції відкидає нульову вихідну розмірність, тож жодна валідно створена `CharRNN` не може дістатися цієї гілки (спроба зламати це призводить до непередбачуваної поведінки — перевірено, тест прибрано). Негативний контроль із вимкненими guard-ами не запускався навмисно: без guard-а кожен такий випадок є UB, тож результат був би оманливим.

Перевірено запуском `main_rnn.cpp`: loss падає 3.30 → 0.44 за 300 епох, генерація працює, `exit=0`. Зміна зачіпає `src/demos/char_rnn.cpp` (виклик OK, але це вперше в цьому етапі змінюється код демо) і `tests/gradcheck.cpp`.

#### Gradient checking MNIST MLP — P2 (продовження)

`tests/gradcheck.cpp` підключає `src/demos/mnist_mlp.cpp`, тобто перевіряється **реальна** композиція з демо, а не копія: `Dense(f1) → ReLU(a1) → Dense(f2) → ReLU(a2) → Dense(f3)`. Щоб уникнути масивного sweep по 235k параметрів, додано конструктор з довільними розмірами `MLP(in, h1, h2, out)` — дефолтний MNIST-конфіг 784/256/128/10 лишився незмінним. Тест використовує крихітну модель 4/6/5/3, batch 3; перевіряються всі 6 груп параметрів і градієнт по входу. Найгірша похибка **9.11e-06**. Тест розташований останнім у `main()`, бо `randomize()`/`random_tensor()` просувають спільний RNG — так попередні результати не зсуваються.

Клас `MLP` не можна було підключити напряму через `#define main ...`: його CLI-частина тягне `utils/mnist.h`, який, своєю чергою, інклюдить відсутній у репо `stb_image.h`. Тому CLI-частину (inline `main` + `mnist.h`) обгорнуто в `#ifndef SHADE_MLP_NO_MAIN`; харнес визначає цей макрос і бачить лише struct `MLP`. Демо-код поза харнесом не змінюється.

Також `MLP::backward` тепер **повертає** `d(input)` замість `void` — виклик у CLI ігнорує повернення, тож це source-сумісно, але дає харнесу змогу перевірити й градієнт по входу (підтверджено verbose-виводом: `param[0..5]` + `input[0]`).

**Явний захист від ReLU-зламу.** ReLU має невизначену похідну в 0: якщо перед-активація потрапляє в зону `±eps`, центральна різниця бачить іншу гілку, ніж аналітичний похідний, і тест падає з причини, не пов'язаної з математикою `backward`. Замість того, щоб покладатися на seed 2026, `prepare_mlp_with_relu_margin()` перегенеровує модель (до 64 спроб), поки **всі** сирі перед-активації f1/f2 не будуть принаймні за `20 * GRAD_EPS = 2.0e-02` від нуля. Важливо: `f1/f2` створені з `use_relu=true`, тож `Dense::forward` сам клямпує значення і не зберігає сиру перед-активацію; `Act::last_in` — вже після ReLU (здебільшого нулі), тому сирі значення обчислюються напряму як `x·W+b` (для f2 — після ReLU від f1). `f3` має `use_relu=false` і захисту не потребує.

`GRAD_EPS = 1e-3` винесено у спільну константу — ним користуються всі три finite-difference харнеси й перевірка мінімальної відстані, тож вони не можуть розійтися. Фактичний мінімальний запас у поточному прогоні: **7.123e-02** (вимога 2.000e-02). Негативний контроль із неможливим порогом (`1e10`) підтвердив, що вичерпання спроб коректно дає `FAIL` та `exit=1`.

#### Gradient checking MNIST CNN — P2 (продовження) + виправлення backward

`tests/gradcheck.cpp` підключає реальний `src/demos/mnist_cnn.cpp` (за тим самим принципом `#ifndef SHADE_CNN_NO_MAIN`) і перевіряє повну композицію: `Conv(3×3,p=1) → ReLU → MaxPool(2×2,s=2) → Conv(3×3,p=1) → ReLU → MaxPool → Dense(ReLU) → Dense`. Додано конфігурований конструктор `CNN(h, w, cin, c1o, c2o, hid, out)`; дефолтний MNIST-конфіг 28×28 / 1→8→16 / 128 / 10 не змінено; помилку в коментарі (`Dense(400→128)`) виправлено на фактичні `784 → 128 → 10`. Тест використовує крихітну модель `6×6, 1→2→3, hid=4, out=3` (6→3→1 просторова, `flat_dim=3`, 108 параметрів), batch 1. Перевірено **усі 8 груп параметрів + градієнт по входу**, найгірша похибка **3.47e-06**.

Перевірка виявила **реальну помилку в демо**: у `CNN::backward` ReLU застосовувався *після* conv backward (`c2.backward(a2.backward(d2))` було навпаки `a2.backward(c2.backward(d2))`). Через це похідні `c1.w/c1.b` і `c2.w/c2.b` обчислювалися без множника ReLU, а `d(input)` — у невірному порядку. Поки результат `a1.backward(...)` у CLI відкидався, навчання «працювало», але градієнти згорткових шарів були хибними. Виправлено порядок: спочатку `a.backward()`, потім `conv.backward()`. Повернення `d(input)` із `CNN::backward` source-сумісне — CLI ігнорує значення.

Як і для MLP, `prepare_cnn_with_relu_margin()` перегенеровує модель (до 64 спроб), поки всі сирі перед-активації (conv1, conv2 та `f1.last_input·W+b`) не будуть за `2.0e-02` від нуля. `a1/a2.last_in` містять сирі conv-виходи (Conv2D не клямпує), тож читаються напряму; сиру перед-активацію `f1` (Dense з `use_relu=true`) обчислено з кешованого `f1.last_input`. Спроба додатково перевіряти запас між двома найбільшими значеннями у вікні MaxPool була відкинута як непотрібна: після виправлення порядку backward тест стабільний і без неї.

#### Gradient checking TransformerLM — P2 (продовження)

`tests/gradcheck.cpp` підключає реальний `src/demos/transformer_lm.cpp` через `#define main transformer_lm_cli_main` (як і `gpt.cpp`) і перевіряє композицію `Embedding → PositionalEncoding → TransformerBlock ×2 → Dense`. Конфіг 8 vocab / dim 4 / 2 блоки / seq 5; 15 груп параметрів (embed.w, output.w/b, 12 параметрів блоків). Входи дискретні, як і в GPT, тож перевіряються лише параметри. Найгірша похибка **1.47e-05**. Показово, що структура практично ідентична `GPT` — тест гарантує, що демо-копія не розійдеться з основною моделлю. `transformer_test.cpp` окремо не перевіряється: там лише `TransformerBlock → Dense`, уже покрите.

**Нормалізація loss у transformer_lm (виправлено).** Як і в CharRNN раніше, навчальний цикл обчислював `dlogits = p - onehot` (градієнт **суми** CE), а друкував `loss / seq_len` (середнє). Додано `TransformerLM::compute_loss`: повертає mean CE і пише `dlogits = (p - onehot)/T`; цикл тепер використовує його напряму (`tl += loss`). Окремий харнес `TransformerLMLoss` звіряє ці `dlogits` зі скінченною різницею скалярного mean-loss — **1.22e-04**. Демо вимагає `data/train.txt`, якого в репо немає, тому запуск завершується повідомленням про відсутній файл (`exit=0`); натомість loss-шлях перевірено gradcheck-ом, а не запуском тренування.

**Усереднення накопичених градієнтів (виправлено після аудиту `bb68358`).** `compute_loss` повертає mean-loss на одне вікно, але після `batch_size = 8` мікробатчів у градієнтах лежала **сума** восьми mean-градієнтів, тоді як друкувався `tl / iters` (середнє). Масштаб оновлення залежав від кількості накопичених вікон, особливо коли `clip_grads` не спрацьовував. Перед `clip_grads()` і `opt.step()` додано ділення на фактичний `iters_since_step` — і для повної групи, і для останньої неповної (не на фіксовані 8). Ділення розташоване **до** clipping, бо кліпування суми й кліпування середнього дають різний результат. Симетрично до виправлення в CharRNN.

**Валідація `TransformerLM::compute_loss` (виправлено там само).** Перенесено guard-и з `CharRNN::compute_loss`: `vocab_size > 0`, ширина logits, непорожня послідовність, `targets.size() == logits.rows`, а raw `f32` target перевіряється **до** `(int)` — `isfinite`, діапазон `0 ≤ raw < vocab_size` і `trunc(raw) == raw`. Новий харнес `TransformerLMVal` відхиляє ті самі 10 видів вводу, що й `CharRNNVal`, на детермінованих даних (без споживання RNG, тому попередні числа не зсунулись). Guard `vocab_size > 0` лишився захисним і, як і в CharRNN, не тестується напряму: `Dense` у head-і відкидає нульову вихідну розмірність ще на конструкції.

#### Gradient checking CrossEntropyLoss::forward — P2 (завершення)

`CrossEntropyLoss` — спільний loss для `main.cpp`, MNIST MLP/CNN; до цього перевірявся лише побічно (simd_check порівнює AVX із scalar-еталоном для `n=3, C=19`). Новий харнес у `tests/gradcheck.cpp` перевіряє його напряму:

- **finite differences** з `eps=1e-3` для `(n,C)` = `(1,8)`, `(3,5)`, `(4,9)`, `(8,16)`, `(2,7)` — тобто `C` кратне 8 (AVX: один/два блоки), `C < 8` (scalar-only) і `C = 9` (AVX-блок + scalar-хвіст); batch 1..8 перевіряє ділення на `n`. Найгірша FD-похибка **1.91e-04** (поріг `2e-3`).
- **незалежний double-еталон** на тих самих f32-входах для всіх кейсів; найгірша похибка **9.50e-08** (поріг `1e-4`). Еталон особливо потрібний для `base=1000` (перевірка віднімання максимуму) і `spread=100` (exp-clamp/underflow): там скінченна різниця f32-loss вимірювала б переважно помилку округлення при відніманні двох великих значень, а не градієнт, тому FD для них пропущено.
- **one-hot targets** рядкової форми `n×C`, як і в усіх реальних викликах.

**Guard-и розмірів у `CrossEntropyLoss::forward` (додано).** Раніше порожній batch ділив на нуль, а target з іншою формою читав поза своїм сховищем за адресою `i*C + j`. Додано `n > 0`, `C > 0` і `target.rows/cols == logits.rows/cols` через `std::invalid_argument`. `CEVal` відхиляє 5 випадків: порожні рядки, нуль класів, розсинхрон рядків/стовпців target і порожній target. Математика loss/градієнта не змінена.

Наступні кандидати — лише P3-нотатки (можливий MaxPool tie-gap guard за іншого seed/config і порівняння `(f32)vocab_size` через `double`), якщо з'явиться підтвердження на даних.

### Ще не зроблено

- Інші шляхи запису (`Serialize::save_model`, `AsyncSaver`) ще не атомарні; GPT checkpoint — уже так.
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
