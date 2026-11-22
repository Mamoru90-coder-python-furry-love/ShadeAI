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
CE n4 C9         OK  ref 1.49e-08  fd 1.15e-04
CE n8 C16        OK  ref 9.50e-08  fd 1.91e-04
CE n2 C7         OK  ref 6.11e-08  fd 6.78e-05
CE bigbase       OK  ref 7.66e-08  fd 8.34e-05 (eps=0.05)
CE wide          OK  ref 9.78e-09  fd 4.08e-05 (eps=0.05)
CEVal            OK
```

### Виправлені математичні помилки

- AdamW: `km = lr*ib1`, `kw = lr*wd`; decay без bias correction.
- Positional encoding: частота залежить від пари координат `(j & ~1) / dim`.
- Cross-entropy: `dlogits` ділиться на batch; у `gpt.cpp` — ще на `T`.
- `CrossEntropyLoss::forward`: loss через log-sum-exp замість `log(max(p, 1e-7f))` — похідна знову відповідає `(p − onehot)/n`, коли p правильного класу << 1e-7.
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

- **finite differences для всіх 7 кейсів**: `(n,C)` = `(1,8)`, `(3,5)`, `(4,9)`, `(8,16)`, `(2,7)` з `eps=1e-3` (AVX: один/два блоки; scalar-only; AVX + scalar-хвіст; batch 1..8 перевіряє `1/n`), а також `base=1000` і `spread=100` з `eps=0.05` — там loss ~60, тож `eps=1e-3` вимірював би переважно округлення f32. Найгірша FD-похибка **1.91e-04** (`CE n8 C16`, поріг `2e-3`); `CE wide` — **4.08e-05**.
- **незалежний double-еталон** на тих самих f32-входах для всіх кейсів: log-sum-exp у double без жодного clamp; найгірша похибка **9.50e-08** (поріг `1e-4`). Раніше еталон повторював `max(p, 1e-7f)` і тому підтверджував хибну реалізацію, а не перевіряв loss ↔ gradient.
- **one-hot targets** рядкової форми `n×C`, як і в усіх реальних викликах.

**Математика loss ↔ gradient (виправлено після аудиту `710f44c`).** Старий `log(max(p, 1e-7f))` робив loss пласким для p < 1e-7, тоді як `dlogits` лишався `(p − onehot)/n` — похідна більше не відповідала повернутому скаляру. У `CE wide` цільовий logit на 48–64 нижче максимуму, тобто p << 1e-7. Loss переписано як `log(sum(exp(z−m))) − (z_t−m)` (для загального target: `-Σ t_j (z_j − m − logsum)`); це стабільно навіть при underflow `exp`. Негативний контроль: зі старою формулою `CE wide FAIL ref 7.13e-01 fd 3.33e-01`, gradcheck `exit=1`; з виправленою — `OK ref 9.78e-09 fd 4.08e-05`.

**Guard-и розмірів у `CrossEntropyLoss::forward` (додано).** Раніше порожній batch ділив на нуль, а target з іншою формою читав поза своїм сховищем за адресою `i*C + j`. Додано `n > 0`, `C > 0` і `target.rows/cols == logits.rows/cols` через `std::invalid_argument`. `CEVal` відхиляє 5 випадків: порожні рядки, нуль класів, розсинхрон рядків/стовпців target і порожній target. Самі guard-и математику не змінюють.

P3-нотатки (MaxPool tie-gap, `(f32)vocab_size`, soft labels) зафіксовано без змін і без коміту: підтверджень на даних немає, а ризик `vocab_size` дає хибне відхилення граничного ID, не OOB.

#### Атомарний запис чекпойнтів і AsyncSaver — P3

`Serialize::save`, `save_model`, `save_optimizer` писали напряму в ціль (`fopen(path, "wb")` одразу обрізає старий файл): переривання лишало урізаний checkpoint. Додано `Serialize::detail::write_atomic()` — запис у сусідній `path + ".tmp"`, `fflush`, `fclose`, потім одна атомарна заміна (`MoveFileExA(..., MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)` / `rename`). Помилка видаляє temp і не чіпає ціль. `save_gpt()` тепер використовує той самий `atomic_replace` замість локальної копії — формат v2 (SHDG + версія + CRC) не змінено.

`AsyncSaver` отримав конфігуровані шляхи (default як раніше), скидання optimizer-стану в `queue_model()`, `join` попереднього потоку в `spawn()` (інакше присвоєння joinable `std::thread` → `std::terminate`), RAII-деструктор `~AsyncSaver() { flush(); }` і перевірку результату публікації з попередженням на stderr.

**P2 (аудит `36a4b3b`) — узгодженість пари model + optimizer.** Дві окремо атомарні публікації не гарантують узгодженості: успішна модель + помилка optimizer лишає нову модель зі старим `t`/m/v, і обидва `load_*` приймають такий мікс. Додано `Serialize::CheckpointPair`: модель і optimizer пишуться як `base.gN` / `opt.gN`, перевіряються, і лише потім атомарно замінюється крихкий маніфест `base.manifest` (magic "PAIR", `gen`, `flags`) — єдине джерело істини. Читач бачить або стару пару, або нову; проміжні `gN`-файли ігноруються. Після публікації збираються всі покоління крім двох останніх. Відсутній маніфест → legacy-fallback на звичайні `base`/`opt`. `CheckpointPair::load` стейджить обидва файли й комітить разом, тому пошкоджений optimizer не оновлює модель частково. Для стейджу `load_model`/`load_optimizer` переведено на спільні `detail::read_model_blocks` / `read_optimizer_blocks` — зовнішня поведінка не змінилась.

**P2 (аудит `d84f0d0`) — синхронізація протоколу.** Додано спільний `pair_detail::io_mutex()` (in-process, не між процесами). `publish()` тримає його весь цикл «читання маніфесту → запис файлів покоління → заміна маніфесту → GC», тому два паралельні publisher-и не можуть захопити одне покоління. `load()` тримає той самий mutex від читання маніфесту до коміту обох тензорів — GC не може видалити вже вибране покоління посеред читання. Контракт: одного mutex-а достатньо для потоків одного процесу; координація між процесами не реалізована.

**P3 (аудит `d84f0d0`).** `read_manifest()` тепер розрізняє `ENOENT`/`ENOTDIR` (справді немає файлу → legacy fallback) від `EACCES` та інших помилок `fopen` (наявний, але недоступний маніфест → жорстка відмова, без fallback). Перевірка `m.size() == v.size()` виконується завжди — `v` без `m` більше не публікується як model-only generation.

Тести в `tests/serialize_checks.cpp`: atomic — успішний запис не лишає `.tmp`; застарілий `.tmp` не впливає на good-файл; заблокований `.tmp` → старі байти й старий model; NaN відхиляється до запису; AsyncSaver — round-trip, подвійний `spawn()`, NaN-снапшот лишає last good. `pair_checks()` — 12 кейсів: g1 round-trip, **модель записана / optimizer заблокований → читач лишається на g1** (плюс доказ, що окремі `load_*` приймають мікс), recovery, корупція маніфесту + `gen=2000000`, отруєний optimizer g3 без часткового commit, GC, model-only generation, **`v` без `m` відхиляється**, **нечитаний маніфест (каталог замість файлу) не падає у legacy**, **8 паралельних publisher-ів → маніфест=`g8`, жодного втраченого покоління**, **64 публікації + reader-раунд → жодного видаленого/змішаного покоління**, legacy fallback.

Негативні контролі (фактично запускались): прямий запис `save_model` → 5 `FAIL`, `exit=1`; прибраний `join` у `spawn()` → `0xC0000409` (`std::terminate`); повернення прямої публікації без маніфеста → `FAIL: publish fails when optimizer temp blocked`, `reader still sees old pair`; **прибраний `io_mutex` → 4 `FAIL`** (`all concurrent publishes succeeded`, `no generation lost`, `last concurrent pair coherent`, `reader never sees deleted or mixed generation`); **прибране розрізнення ENOENT → `FAIL: unreadable manifest blocks legacy fallback`**; **повернута стара перевірка m/v → `FAIL: v without m rejected`**. Усі три останні: `pair_checks: FAILED`, після відновлення — `OK`.

### Маркування `shade.exe` — VERSIONINFO

Окремим комітом (не змішується з виправленнями ядра) додано `src/shade.rc` із `VS_VERSION_INFO`: `CompanyName=ShadeFurry Technologies`, `ProductName=ShadeAI`, `FileDescription=ShadeAI Application`, `LegalCopyright=ShadeFurry Technologies`, `FileVersion/ProductVersion=1.0.0.0`, `OriginalFilename=shade.exe`. `build\compile.bat` тепер вимагає `cl` **і** `rc` (обидва з `vcvars64.bat`), збирає `build\shade.res` і лінкує його в `shade.exe`; `*.res` додано до `.gitignore`. Після ініціалізації середовища повторно перевіряються `vswhere`/`vcvars64.bat`/`cl`/`rc` — у разі невдачі скрипт падає зі зрозумілим повідомленням замість плутаної помилки `rc` (P3-нотатка аудиту `fdab597`). Негативний контроль: `VSPATH=C:\bogus-vs-path` → `vcvars64.bat not found`, `exit=1`. Фактично перевірено через `(Get-Item build\shade.exe).VersionInfo` — усі поля зчитуються.

**Це метадані, а не Authenticode.** Windows не вважає файл криптографічно підписаним. Для справжнього підпису потрібен code-signing сертифікат і `signtool sign`; цей крок не виконано.

#### Аудит BPE / токенайзера (база `516250c`)

Новий харнес `tests\tokenizer_checks.cpp` (підключено до `build\test.bat`) — 12 кейсів, які спочатку запускались на **незміненому** коді й дали 11 `FAIL` (негативні контролі), а після виправлень — `OK`.

**`BPETokenizer::load` (був найнебезпечнішим):**
- `std::stoi` без `try/catch` → виняток вилітав із `load()` на рядку типу `token abc`;
- `id2token.resize(max_id + 1)` без меж → `id=2000000` виділяв мільйони рядків; тепер `MAX_ID=1000000`;
- негативні ID лишались у `vocab`, але не потрапляли в `id2token`;
- дубльовані ID тихо перезаписували `id2token` у hash-порядку → недетермінований decode;
- дубльовані ключі (`SPACE 6` і `  6`) тихо перезаписувались;
- повторний `load()` не чистив стан — старі токени лишались у `vocab`;
- частково пошкоджений вокабулар лишав напівзавантажений стан після помилки.

Тепер парсинг іде в тимчасові контейнери й комітиться тільки після успішного читання **обох** файлів; перевіряються унікальність ключів та ID, діапазон ID, наявність усіх чотирьох special-токенів. Перший виграш дубля merge детермінований (`emplace`, а не `operator[]`).

**P2 аудиту `26fccf4`:** непорожній рядок вокабуляра без роздільника ID (`corrupted_entry`) раніше тихо пропускався — пошкоджений файл міг виглядати валідним із втраченим записом. Тепер такий рядок → жорстка відмова. Те саме для рядка merges без двох операндів (або з зайвим). Порожні рядки лишаються допустимими — саме їх породжує експорт токена `"\n"` у `train_bpe.py`. Додано перевірки `fvocab.bad()` / `fmerges.bad()`, щоб помилка вводу-виводу не перетворювалась на успішне часткове завантаження. Негативний контроль із версією `26fccf4` (тимчасово повернутою з git): 4 `FAIL` (`vocab line without id`, `corrupt vocab does not commit partial state`, `malformed merge line`, `corrupt merges do not commit partial state`), після відновлення — `OK`.

**Сумісність із `train_bpe.py`:** Python-експортер пише токен `"\n"` як буквальний новий рядок + `" <id>"`, тож `getline` повертав порожній токен, і `\n` зникав з `encode()`. Порожній токен тепер інтерпретується як `"\n"` (залишено і явні аліаси `SPACE`/`NEWLINE`). `BPETokenizer::tokenize()` переписаний за тими самими правилами пробілів/нового рядка, що й `encode()` (раніше `istringstream >>` просто викидав новий рядок).

**`Tokenizer` (SHDG-чекпоінти):**
- `build()` тепер сортує за частотою **за спаданням, потім за токеном** — `unordered_map` дає нестабільний порядок для однакових частот, що робило ID токенів залежними від STL-реалізації;
- `read()` відхиляє порожній токен (`len <= 0`) — він не може бути створений `tokenize()` і робив би `decode()` двозначним.

Не виправлено (P3, без доказів впливу): `utf8_chars()` не валідує байти UTF-8 (зараз це byte-level fallback); BPE не має власної бінарної серіалізації в SHDG — він читає текстові `vocab/merges`.

#### Аудит AdamW (база `897c27b`)

Новий харнес `tests\adamw_checks.cpp` (підключено до `build\test.bat`) порівнює ядро з **незалежним еталоном у `double`** (`AdamWRef`) на 4 кейсах. На незміненому коді: 2 `FAIL`, після виправлення — 4 `OK`.

**P1 — неправильний порядок decoupled weight decay.** Було `np = (p - step) * (1 - lr*wd)`: затухання застосовувалось до вже оновленого параметра, тож адаптивний крок випадково множився на `(1 - lr*wd)`. За формулою AdamW (Loshchilov & Hutter) і PyTorch треба `p_{t+1} = p_t * (1 - lr*wd) - lr * m_hat / (sqrt(v_hat) + eps)` — decay стосується **старого** `p_t`. Негативний контроль (тимчасово повернуто `adamw.h` із `897c27b`):

```text
adamw_analytic_1step: FAIL (worst |got-ref|=0.0500001, lr=0.5/wd=0.2)
adamw_multistep:      FAIL (worst |got-ref|=0.00829428, lr=0.05/wd=0.3)
adamw_zero_grad_decay: OK   // регресійний контроль
adamw_no_decay:        OK   // регресійний контроль
```

Виправлено в обох гілках — AVX (`np = p - kw*p` перед відніманням кроку) й скалярній. GPT у `train_gpt` використовує `wd=0`, тож його поведінка не змінилась; MLP/CNN/TransformerLM (`wd>0`) тепер відповідають еталону.

**P2 — хибний «незалежний» еталон.** У `tests\simd_check.cpp` референс теж робив `e -= lr*wd*e` після кроку, тобто відтворював той самий баг і не міг його виявити. Виправлено на `e = pref - lr*wd*pref; e -= lr*m_hat/(...)`.

**P3 — зафіксовано, не змінював:** стан оптимізатора (`t`, `m`, `v`) не серіалізується у SHDG v2; `train_gpt` починає з нульових моментів після перезавантаження чекпойнта. `AsyncSaver` з SHDE-generation існує в `serialize.h`, але не викликається продакшн-шляхом. Перевірок розмірів `params/grads/m/v` і обробки NaN/Inf-градієнтів немає (наразі стримується `clip_grads(1.0f)` у GPT).

#### Аудит optimizer checkpoint / безперервність тренування (база `652dacb`)

Новий харнес `tests\optimizer_resume_checks.cpp` (підключено до `build\test.bat`): детерміновані 10 кроків AdamW порівнюються з 4 кроками → checkpoint → 6 кроків. Після коректного відновлення `t/m/v` розбіжність **рівно 0** (`after load: |dp|=|dm|=|dv|=0, t=4`). Негативний контроль (відновлено тільки ваги, моменти скинуто) дає `worst |diff|=0.153197` — тест чутливий до втрати моментів.

**P1 — legacy-шлях `CheckpointPair::load` тихо деградував при пошкодженому sidecar.** Було: якщо `fopen(opt_path)` не вдався або `read_optimizer_blocks()` повернув false, код просто ставив `has_opt=false`, комітив ваги й повертав `valid=true`. Тобто пошкоджений або несумісний файл моментів виглядав як «model-only чекпойнт», і тренування продовжувалось з нульовими моментами та скинутим `t`. Generation-шлях (з маніфестом) уже падав у таких випадках — legacy тепер узгоджений з ним.

Негативний контроль (старий `serialize.h` із `652dacb`, тимчасово повернутий із git):

```text
FAIL: corrupt legacy optimizer fails the whole load
FAIL: corrupt legacy optimizer commits nothing
FAIL: incompatible optimizer shape rejected
optimizer_resume_checks: FAILED
```

Після виправлення: `OK`. Розрізнення: відсутній файл (`ENOENT`) = легітимний model-only чекпойнт; наявний, але пошкоджений/нечитаний/несумісний = жорстка відмова без коміту ваг чи моментів.

**P3-A / P3-B аудиту `ff8e4a7` — закрито окремими комітами (`97f06b0`, потім уточнення).** `m.size() != v.size()` спочатку перенесли перед `if (!m.empty())` у legacy-гілці, але аудит `97f06b0` показав, що generation-гілка з model-only маніфестом (`FLAG_HAS_OPT=0`) лишала перевірку осторонь. Фінально перевірку перенесено на **спільний рівень `CheckpointPair::load()`** — до розгалуження legacy/generation; у generation-гілці лишилась окрема відмова `m.empty()` при `has_opt=true`. Негативний контроль проти `97f06b0`: `FAIL: v without m rejected on generation model-only load`; після виправлення — `OK`. Додано два кейси: legacy model-only і generation model-only з `v` без `m`.

`ENOTDIR` прибрано зі списку «файл відсутній» у `read_manifest()` і legacy-гілці `load()`. Проба `fopen()` на MSVC показала: absent → `ENOENT=2`, каталог → `EACCES=13`, файл-компонент шляху → `ENOENT=2`, trailing separator → `EINVAL=22`; `ENOTDIR` CRT не повертає взагалі. Тому зміна policy на POSIX запобігає прихованій помилці конфігурації, а на Windows не змінює поведінку — `EACCES`/`EINVAL` і так були жорсткими відмовами.

**Помічено, не виправлено (P3):** `sz/2` для поточних тестових тензорів потрапляє в заголовок першого `v`-тензора — коментар у тесті виправлено на точний. Отже контроль перевіряє відмову на некоректній **структурі**, а не на зміненому float. Довільна зміна валідного float у payload лишається непомітною — sidecar не має контрольної суми (на відміну від SHDG v2 із CRC). Для повного захисту потрібна версіонована CRC у форматі optimizer-файлу; без доказів впливу формат не змінювався.

**P3 — API-контракт, зафіксовано:** `CheckpointPair::load` повертає `Result::t`, але не записує його в `AdamW::t` — виклична сторона має зробити `opt.t = res.t` явно. Перша версія харнесу саме це пропустила: після load `res.t==4`, а на першому ж кроці `t` став 1 (помилка тесту, не продакшну; після виправлення розбіжність 0). Варто додати `AdamW::restore(...)`-хелпер, щоб виключити таку помилку в майбутньому.

**P3 — залишається відкритим:** `train_gpt` досі не має CLI `--resume` і не пише optimizer-сайдкар; `save_gpt` зберігає лише SHDG v2 (ваги + токенайзер). Отже описаний вище механізм коректний, але продакшн-шлях ним не користується — для реального resume треба окремий коміт із зміною CLI/формату.

#### Аудит AdamW edge cases (база `da9b8c7`)

Новий харнес `tests\adamw_edge_checks.cpp` (підключено до `build\test.bat` із `%CFLAGS%`, тобто з `/arch:AVX`). Незалежний еталон — `double`-реалізація канонічної формули з окремою bias-correction для `m` і `v`; продакшн-результат порівнюється elementwise. Покриття: **24 кейси** — межі `lr` (1e-8 … 1.0, від'ємний), `wd` (0 … 3.0, від'ємний, `lr*wd>1`), `beta1/beta2` (0, 0.5/0.9, 1.0), `eps` (1e-8, 1e-3, 0), нульові/1e-20/1e6-градієнти, 2000 оновлень, n=37 (32 AVX + 5 scalar tail), NaN/Inf у градієнті, moment і параметрі, попередньо обрізані градієнти → decay. Уточнення аудиту `cabaf64`: кейс із обрізанням масштабує градієнти вручну за формулою `GPT::clip_grads(1.0f)` — він перевіряє AdamW на вже обрізаних градієнтах, **а не виклик `clip_grads` чи його інтеграцію**. Харнес успішно будується й проходить і з `/arch:AVX`, і без нього (scalar-only).

**Результат: дефектів у продакшн-коді не знайдено, код `src\optimizer\adamw.h` не змінювався.** Усі 24 кейси: `worst` у межах 4.6e-7 … 1.9e-5 за заданих tolerances; `prodNaN == refNaN` і `finDiff=0` скрізь (порівняння скінченності ведеться **по кожному індексу**, а не лише «є десь NaN»). Випадки `beta1=1`, `beta2=1`, `eps=0 при g=0` дають NaN і в еталоні, і в продакшні — це властивість самої формули (ділення на 0 / `1-1`), а не розбіжність реалізації.

**Чутливість харнесу перевірено мутаційним контролем:** тимчасово повернуто старий порядок decay (спочатку крок, потім `p -= kw*p`) в обох гілках — 6 кейсів упали (`wd_large_1`, `wd_negative`, `beta_low`, `eps_large_1e-3`, `avx_plus_scalar_tail`, `preclipped_then_decay`), потім файл відновлено з резерву й `git diff --quiet -- src/optimizer/adamw.h` підтвердив чистоту. Тобто тест справді детектує порушення саме тієї математики, яку аудитуємо.

**P3 — пастка при написанні еталона (зафіксовано в коментарях):** `EdgeRef` приймає `(lr,b1,b2,eps,wd)`, а `AdamW` — `(lr,wd)` + поля. Перша версія харнесу в ручних блоках переплутала порядок (`eps` став 0.999), через що 4 кейси хибно падали з розбіжністю ≈ одному кроку. Після виправлення — `OK`. Це ще раз підтверджує, чому еталон мусить бути незалежним і перевіреним.

**P3 — локалізація NaN/Inf:** кожен елемент оновлюється незалежно; отруєний `g`/`m`/`p` не поширюється на сусідні ваги (перевірено чотирма окремими тестами з поіндексовою перевіркою `isfinite`). Крім того, для `inf_gradient_is_local` і `nan_moment_is_local` додано явні assertions, що саме отруєний індекс став NaN в обох — продакшні й еталоні. Це не означає, що NaN «обробляється» — він лишається у своєму елементі; політики sanitation/сканування немає.

#### Аудит контракту розмірів `AdamW::step` (база `cabaf64`)

Новий харнес `tests\adamw_contract_checks.cpp` (підключено до `build\test.bat`). Негативні контроли побудовані так, щоб на **невиправленому** коді викликати лише in-bounds мутацію — жодного читання/запису поза vector. Це дозволило довести дефект executable-тестом, а не лише оглядом коду.

Негативний контроль на `cabaf64` (8 хибних наборів, усі `FAIL (t 0->1)`):
```
extra grads / m / v tensor
grads[0] / m[0] / v[0] larger than params[0]
grads[0] shape 4x2 vs params 2x4
grads[1] invalid of 3
```
Валідний контроль (`valid pair updates once`) на старому коді проходив — `p 1→0.89, t=1`. Тобто харнес не «завжди падає».

**P1 — `AdamW::step` індексував `grads[i]`, `m[i]`, `v[i]` за `params.size()` без перевірки.** Надлишкові тензори тихо ігнорувались, а короткі — давали UB. Виправлення: єдина перевірка на початку `step()`, **до `t++` і до будь-якої мутації**:

- `params.size() == grads.size() == m.size() == v.size()`, `params` не порожній;
- кожен `params[i]`/`grads[i]` ненульовий;
- `rows`/`cols` збігаються для `params[i]` ↔ `grads[i]`, `params[i]` ↔ `m[i]`, `params[i]` ↔ `v[i]`.

Недійсний набір = no-op: жоден параметр, момент і `t` не змінюються. Після фіксу 8 негативних контролів дають `OK (t 0->0)`; додано 6 regression-кейсів із **короткою** стороною (`missing grads/m/v`, `grads[0]/m[0]/v[0] smaller`) — на старому коді вони були б UB, тепер перевіряються безпечно.

**P2 аудиту `63e973c` — `data.size()` міг розійтися з `rows/cols`.** Усі поля `Tensor` публічні, тож `g.data.resize(4)` не змінює `1×8` у метаданих. Перевірка `rows/cols` проходила, а ядро ітерує `params[i]->size()` = `data.size()` — короткий `grads/m/v` давав би UB. Негативні контроли на `63e973c` (напрямки обрані так, щоб старий код лишався in-bounds: `params` — короткий, інші три — довгі):

```
params data.size() 4 vs 8    FAIL (t 0->1)
grads data.size() 12 vs 8    FAIL (t 0->1)
m data.size() 12 vs 8        FAIL (t 0->1)
v data.size() 12 vs 8        FAIL (t 0->1)
```

Виправлення в тій самій передмутаційній перевірці: для кожного з чотирьох тензорів `data.size() == rows*cols` (з від'ємними `rows/cols` — відмова) **і** `g/m/v.data.size() == p.data.size()`. Після фіксу всі 4 → `OK (t 0->0)`, валідний контроль → `OK (p 1→0.89, t=1)`. Разом **18 хибних наборів + 1 валідний контроль**.

**P3 — сигнал про відмову:** `step()` лишається `void`; виклична сторона не дізнається про відмову, окрім відсутності змін стану. Зміна сигнатури на `bool` не потрібна для поточних викликів (усі ігнорують повернення), але може бути додана окремо, якщо з'явиться потреба в обробці помилки.

#### Аудит `--resume` у `train_gpt` (база `3ae4cbe`)

Негативний контроль на невиправленому коді: `shade.exe --resume <model.bin> hello` → `Unknown argument: --resume`, exit=1. `train_gpt` завжди починав з `ep=0`, `opt.t=0`, `srand(42)` і зберігав лише SHDG v2 (ваги + конфіг + токенайзер). `Serialize::CheckpointPair` не можна перевикористати напряму: він пише **SHDE** через `save_model()`, а GPT — **SHDG** з власним CRC і токенайзером. Механізм `CheckpointPair` не змінювався.

**P3 (було зафіксовано, пізніше виправлено в audit `d8ca469`):** `Tensor::size()` = `(i32)data.size()` (`src/core/tensor.h:90`). Перевірка `data.size() == rows*cols` у `AdamW::step` не обмежує розмір значенням `INT32_MAX`; вручну розширений `data` > 2^31 елементів все ще може дати хибне `n` у ядрі.

**Обраний формат — SHDG v3, один файл.** Після блоків ваг додається секція resume: `epoch, rand_seed, rand_calls, has_opt` і, за наявності, `n`, блоки `m`, блоки `v`, `t`. Новий запис без resume-об'єкта й досі пише v2. Читання v1/v2 для inference збережено; явний resume v1/v2 відхиляє (не перетворюється тихо на «свіжий старт»). CRC v2/v3 покриває й нову секцію.

**Відновлення RNG — без зміни поведінки свіжого запуску.** Усі виклики `rand()` на шляху тренування (`sample`, shuffle) загорнуто в `shade_rand()` з лічильником. Важлива деталь, знайдена мутаційним контролем: конструктори `Dense`/`Embedding`/`SelfAttention` теж викликають `rand()`, але не враховуються. Тому на resume спочатку `shade_srand(seed)` **до** побудови `GPT`, а replay `rand_calls` — **після**; інакше стан CRT зміщується на N викликів конструкторів. З фіксованим порядком `3 епохи` і `2 епохи + resume` дають **побітово ідентичні** чекпойнти.

**Транзакційність:** `read_gpt_checkpoint` збирає ваги, moments і прогрес у локальні staged-об'єкти й призначає їх викличній стороні лише після CRC, структури, `at_eof` і перевірки NaN/Inf у `m/v`. Відмова = жодної мутації ваг, токенайзера чи стану. `train_gpt` додатково перевіряє кількість/форми тензорів і діапазон епохи перед застосуванням. Чекпойнт пишеться **після кожної епохи**, а не лише після останньої.

**CLI:** додано `--resume [model.bin]` і псевдонім `--continue` (старі запускні скрипти). Обидва вимагають v3; `--load` лишається inference-only.

Новий харнес `tests\gpt_resume_checks.cpp` (у `build\test.bat`, з `%CFLAGS%`): побітовий continuity, зчитування прогресу, CRC-пошкодження секції, чотири structural-відмови з **перерахованим валідним CRC** (`has_opt=2`, `epoch<0`, `rand_calls>INT_MAX/10`, хибні `n`/форма/`t`), обрізаний хвіст, чутливість до скинутих `m/v`, сумісність v2 для inference й відмова v2 для resume. Негативний контроль: прибраний `shade_replay_rand()` ламає побітовий continuity (`worst|dw|=0.0028`), файл відновлено з резерву.

#### Повторний аудит `resume` (база `44ee899`)

Негативні контролі на невиправленому коді: `read_gpt_checkpoint(..., &resume)` приймав SHDG v3 з `has_opt=0` (відмова відбувалась лише глибше в `train_gpt`); CLI повертав `exit=0` для відсутнього/пошкодженого/v3-bез-оптимізатора чекпойнта; завершений чекпойнт без `--data` падав через відсутній у репо `data/tinystories_valid.txt`, бо перевірка епохи відбувалась після читання датасету.

Виправлено трьома точковими змінами в `src/gpt.cpp`:

1. Явний resume відхиляє `has_opt=0` **на межі reader**; inference без resume-вказівника й досі читає такий файл.
2. `train_gpt`/`infer_gpt` повертають `bool`; `main` повертає `1` на будь-якій контрольованій відмові й `0` тільки після успіху. Помилка читання датасету, хибна форма тензорів, епоха поза діапазоном і невдале збереження тепер теж дають `exit=1`.
3. Перевірка `epoch` перенесена перед відкриттям датасету: завершений чекпойнт і епоха > ліміту не вимагають тексту й не читають зайвий файл.

Результати шести напрямків аудиту:

1. **CLI й шляхи.** `--resume [model.bin]`, `--continue`, `--data`, `--model` і шляхи з пробілами працюють; `--model` перевизначає раніше вказаний шлях. `--load` лишається inference-only.
2. **Відсутній/пошкоджений чекпойнт.** Відмова з `exit=1`, без мутації ваг/токенайзера/моментів (транзакційний staged-read).
3. **Датасет і параметри.** Fingerprint датасету у v3 немає: `--resume` з іншим `--data` свідомо продовжує навчання на новому тексті, але токенайзер і архітектура беруться з чекпойнта. Це залишено як задокументований fine-tuning path, не помилка. `epochs` нижче збереженої епохи відхиляється й не змінює файл; вища — продовжує. `dim/blocks/max_len`, передані API/клієнтом, на resume ігноруються на користь чекпойнта. `batch_size` теж серіалізується в v3 і на resume перевизначає передане API значення (див. аудит нижче).
4. **RNG replay.** Межа `rand_calls ∈ [0, 100000000]` перевіряється reader; Continuity `3 епохи` == `2 + resume` лишається побітовою. Граничне значення приймається структурно, але 100M replay не виконується в тестах (окремий P3 про вартість replay).
5. **`has_opt=0`.** Inference приймає, явний resume відхиляє одразу в reader. Додано regression-кейс для обох гілок.
6. **Переривання запису.** Stale `.tmp` споживається atomic replace; якщо tmp-шлях заблокований, `save_gpt` падає, а попередній валідний файл лишається байт-у-байт (існуючий `gpt_checkpoint_check`), для v3 додано окремий stale-tmp контроль. Гарантія — атомарна видимість для crash процесу, не power-loss durability без `fsync`.

`tests\gpt_resume_checks.cpp` розширено: CLI exit-коди, `--model`/`--data`/шляхи з пробілами, завершений чекпойнт без датасету, зміна датасету, зміна ліміту епох, архітектурний override і межа `rand_calls`.

Верифікація після фіксу: `build\test.bat` → `All tests passed.`; `build\compile.bat` → `exit=0`; E2E на `build\shade.exe`: train `exit=0`, inference v3 `exit=0`, inference v1-фікстури `exit=0`, resume завершеного чекпойнта `exit=0`, а відсутній/пошкоджений/v1-як-resume `exit=1`.

#### Аудит `INT32_MAX` у `Tensor::size()` / `AdamW` (база `d8ca469`)

Негативний контроль на невиправленому коді: `(size_t)INT32_MAX + 1` у виразі `(i32)data.size()` дає `-2147483648`; отже всі цикли `for (i32 i = 0; i < size(); i++)` просто не виконуються замість відмови. `AdamW::step` перевіряв `data.size() == rows*cols` у `size_t`, але не мав явної межі `INT32_MAX`, тож міг дійти до ядра з хибним `n` і вже після `t++`.

Виправлення без зміни математики:

- `src/core/tensor.h`: додано `fits_i32(size_t)` і `checked_data_size(size_t)`; `Tensor::size()` тепер кидає `std::length_error` замість тихого переповнення.
- `src\optimizer\adamw.h`: у передмутаційній перевірці додано `if (!fits_i32(want)) return;` — ще до `t++` і до будь-якої зміни параметрів/моментів.

Конструктори вже мали `checked_size()`, тож шлях з валідної форми лишається незмінним. Новий захист критичний саме для публічного `data`, який можна `resize()` без зміни `rows/cols`.

Тести: `tensor_checks` перевіряє межі `INT32_MAX`, `INT32_MAX+1`, `SIZE_MAX`, кидок для надмірного count і незмінність звичайного `size()`; `adamw_contract_checks` додає окремий guard-кейс. Реального виділення >2^31 елементів (8 ГБ) не робилось — перевіряється саме перетворення лічильника. Мутаційний контроль: тимчасовий `fits_i32() → true` дає `tensor_checks: FAILED`; після відновлення — `OK`.

#### Аудит NaN/Inf на межі GPT (база `616f09f`)

Негативні контролі на невиправленому коді:

- NaN-градієнт лишався після `clip_grads`, Inf перетворювався на `NaN (ind)`; повернення `void` не давало сигналу викличній стороні.
- `compute_loss` з NaN-логітом повертав скінченну втрату (`6.45834` → `23.0259`), але отруйний `dlogits`; отже `tl += loss` не виявляв проблеми.
- `sample` з NaN-логітом тихо повертав останній токен.
- `save_gpt` з NaN-вагою повертав `true` і публікував файл, який власний reader потім відхиляв (`wrote=1 reader_accepts=0`).

Виправлення без зміни формул і без зміни `AdamW`:

- `GPT::compute_loss`: явна перевірка форми, скінченності логітів/таргетів і діапазону індексу **до** `(int)` та exp/log; нечисловий вхід → `quiet_NaN()`.
- `GPT::clip_grads`: тепер `bool`, відхиляє нечислові градієнти й нечислову норму **до** `opt.step`.
- `GPT::sample`: нечислові логіти/температура → `-1`; `generate` тепер `bool` із вихідним рядком і передає відмову в `train_gpt`/`infer_gpt` (CLI → `exit=1`).
- `train_gpt`: перевіряє loss і результат `clip_grads`; за відмови не робить крок, не генерує й не зберігає чекпойнт.
- `save_gpt`: відхиляє NaN/Inf/вибухлі ваги ще до відкриття `.tmp`; reader і writer тепер мають однакову політику.

`AdamW` не змінювався навмисно: `tests\adamw_edge_checks.cpp` фіксує локалізацію NaN/Inf як перевірену властивість ядра, а отруйні градієнти тепер відсікаються на межі GPT до виклику оптимізатора. `CrossEntropyLoss` у `src/core/loss.h` теж не змінювався: негативний контроль показав, що NaN-логіт дає NaN loss і NaN-градієнт (сигнал не губиться), а додатковий guard був би поза межею підтвердженого дефекту.

`tests\gpt_checkpoint_check.cpp` додає блок перевірок для всіх чотирьох меж. Мутаційні контролі: окреме вимкнення guard-ів для `clip_grads`, логітів loss, `sample` і `save_gpt` дає відповідні `FAIL`; після відновлення — `OK`. Верифікація: `build\test.bat` → `All tests passed.`; `build\compile.bat` → `exit=0`; E2E train/infer/resume → `exit=0`.

#### Аудит `CheckpointPair::t` / лічильника кроків (база `4f979a6`)

Негативні контролі на невиправленому коді:

- `AdamW::step` із `t=INT32_MAX` давав `t=-2147483648` і `p=0.99` — адаптивна частина оновлення тихо зникала; із `t=-1` давав `t=0` і `p=-nan(ind)`.
- `save_optimizer` приймав `t=INT32_MAX` і повертав його назад; `CheckpointPair::publish` публікував generation із таким `t`.
- `save_gpt` приймав v3 resume з `t=INT32_MAX`, reader його теж приймав. Після одного resume-epoch `opt.t` ставав від'ємним, а наступний reader відхиляв щойно записаний чекпойнт.
- Забуття `opt.t = res.t` при коректних `m/v` дає розбіжність `worst|dw|=0.0298598` — контракт `Result::t` чутливий, але лишається явним.

Виправлення без зміни формул і без переписування протоколу:

- `src\optimizer\adamw.h`: додано два діапазони — `adamw_t_valid()` (`0 <= t < INT32_MAX`, збережуване значення) і `adamw_t_can_step()` (`0 <= t < INT32_MAX - 1`, стан, з якого можна зробити ще один крок). `step()` перевіряє другий **до** `t++` і до будь-якої мутації параметрів/моментів.
- `src\utils\serialize.h`: `save_optimizer`, `read_optimizer_blocks` і `CheckpointPair::publish` використовують `adamw_t_valid()`; `CheckpointPair::load` як API навчального resume додатково вимагає `adamw_t_can_step()`.
- `src\gpt.cpp`: `save_gpt` і reader SHDG v3 перевіряють збережуваність; reader з явним `resume` додатково відхиляє термінальний `t`, який не можна продовжити.
- `CheckpointPair::Result::t` і надалі не призначається автоматично; у `optimizer_resume_checks` додано окремий негативний контроль на забутий `opt.t = res.t`.

Тести: `adamw_contract_checks` додає межі `t`; `optimizer_resume_checks` — відмову `publish`/legacy-sidecar для `t=INT32_MAX` і забутий `res.t`; `gpt_resume_checks` — structural poke `t=INT32_MAX` із перерахованим CRC і відмову `save_gpt`. Кожен structural poke тепер починається з чистої копії чекпойнта — інакше пізніший випадок відхилявся б через попереднє пошкодження, не перевіряючи власний guard.

Мутаційні контролі: вимкнення guard-а в `step()` дає `t=-1→0`, `t=INT32_MAX→INT32_MIN` і `adamw_contract_checks: FAILED`; `adamw_t_valid() → true` ламає `optimizer_resume_checks` і `gpt_resume_checks` (`FAIL` для publish/save/reader на `t=MAX`). Після відновлення — `OK`.

Верифікація після фіксу: `build\test.bat` → `All tests passed.`; `build\compile.bat` → `exit=0`; E2E на `build\shade.exe`: train `exit=0`, inference в3 `exit=0`, resume завершеного чекпойнта `exit=0`.

##### Допрацювання після аудиту `e5bafb6` (прийнята база лишається `4f979a6`)

Негативний контроль на невиправленому `e5bafb6`: `AdamW::step` із `t=INT32_MAX-1` змінював параметри й моменти, але лишав `t=INT32_MAX`, який тоді відхиляли `save_optimizer`/`save_gpt`. Водночас SHDG v3 із `t=INT32_MAX-1` приймався явним resume, хоча жоден наступний крок не міг би зберегти результат.

Рішення — розділити два контракти замість одного guard-а: збережуване значення (`adamw_t_valid`) і вхід для наступного кроку (`adamw_t_can_step`). `step()` відхиляє `INT32_MAX-1` до мутацій; `publish`/`save_gpt`/reader без явного resume лишають термінальний стан читаним для inference; `CheckpointPair::load` і SHDG reader із явним `resume` його відхиляють. Останній дозволений крок із `INT32_MAX-2` лишає `t=INT32_MAX-1`, який усе ще можна зберегти.

Тести додають: `t=INT32_MAX-1` як no-op у `adamw_contract_checks`, контроль «останній крок лишає збережуване t», publish/raw-sidecar для термінального `t` і відмову `CheckpointPair::load`, а для SHDG v3 — успішний запис/читання для inference разом із відмовою явного resume.

#### Аудит `batch_size` у SHDG v3 (база `93a249a`)

Негативний контроль на невиправленому коді: той самий 2-епоховий чекпойнт, продовжений із `batch_size=2`, давав `identical=1, worst|dw|=0`; продовжений із `batch_size=3` — `identical=0, worst|dw|=0.00338719785`. Отже параметр керування навчанням, якого немає у файлі, тихо ламав відтворюваність.

Виправлення без зміни формату v1/v2 і без нового version number:

- `GPTResume` отримав `batch_size` зі значенням за замовчуванням 8 (історичний дефолт).
- `save_gpt` додає один `i32` у кінець resume-секції v3, після `t`; CRC покриває нове поле.
- `read_gpt_checkpoint` приймає як нові файли, так і старі v3, що закінчуються одразу після `t`: у старих лишається дефолт 8. Неповне 1–3-байтове розширення й некоректне значення відхиляються.
- Єдиний валідатор `gpt_batch_size_valid()` (`1..1000000`) використовують і writer, і reader, і свіжий запуск.
- `train_gpt` на resume бере `batch_size` з чекпойнта; значення API ігнорується так само, як `dim/blocks/max_len`. Свіжий запуск із `batch_size <= 0` відхиляється.

Тести в `gpt_resume_checks`: побітова тотожність при resume зі зміненим `batch_size=3` (використано незайманий epoch-2 `twoonly`; `part` до того моменту вже перезаписаний resume до epoch 3), structural poke `batch_size=0` із перерахованим CRC, відмова `save_gpt` і свіжого запуску для `0`, legacy-v3 контроль (відрізати останні 4 байти + перепечатати CRC → дефолт 8). Тест обрізання resume-секції тепер ріже 8 байтів, бо останні 4 є валідним легасі-полем.

Мутаційні контролі: (1) writer не пише поле → ламаються continuity й offset-тести; (2) `train_gpt` бере `tcfg.batch_size` замість збереженого → `stored batch_size beats caller override: FAIL`; (3) `gpt_batch_size_valid() → true` → `FAIL` для reader/writer/fresh-run. Після відновлення — `OK`. Окремий контроль (2) спершу не спрацював саме через помилку staging у тесті — це знайдено й виправлено до прийняття рішення.

Верифікація після фіксу: `build\test.bat` → `All tests passed.`; `build\compile.bat` → `exit=0`; E2E на `build\shade.exe`: train `exit=0`, inference в3 `exit=0`, resume завершеного чекпойнта `exit=0`. Розмір E2E-чекпойнта зріс рівно на 4 байти — нове поле присутнє у фактичному файлі.

**Застереження сумісності (піднято під час аудиту `7c56011`).** Старі v3-файли, створені через API з нестандартним `batch_size`, не зберігали це значення й відновити його заднім числом неможливо. Вони отримують `GPT_DEFAULT_BATCH_SIZE = 8`; якщо оригінальний запуск мав інше значення, подальша траєкторія відрізнятиметься. Нові файли зберігають фактичне значення.

#### Аудит межі `rand_calls` (база `7c56011`)

Негативні контролі на невиправленому коді:

- `save_gpt(rand_calls=150000000)` → `0`: запуск, що спожив понад бюджет replay, не міг опублікувати чекпойнт.
- Чекпойнт із `rand_calls=GPT_MAX_RAND_REPLAY`: `resume=0`, `file_unchanged=1` — 100M викликів `rand()` відгравалися, епоха в пам'яті виконувалася, але запис нового чекпойнта падав через перевищення того самого ліміту.

Рішення — окремі діапазони, як для `t`:

- `gpt_rand_calls_valid()` = `0..INT32_MAX` — збережуване значення. writer і reader приймають його, тож довгий запуск завжди може зберегти чекпойнт.
- `gpt_rand_calls_replayable()` = `0..100000000` — явний resume відхиляє більше значення одразу в reader, не витрачаючи час на replay.
- `shade_replay_rand()` використовує той самий replayable-helper; перед записом додано явну перевірку, що `g_rand_calls` ще влазить в `i32`.

Тести: межа `100M` приймається для inference й явного resume; `100M+1` зберігається, читається для inference, але відхиляється для resume; від'ємне значення відхиляється writer-ом. Повний динамічний контроль із 100M replay виконано окремо (близько 2с), у щоденний набір його не додано.

#### Аудит симетрії конфігурації writer/reader (база `3e9fef6`)

Негативні контролі на невиправленому коді (окремий харнес із `#include "../src/gpt.cpp"`):

- `GPT(..., blocks=0)` → `save_gpt=1`, `reader=0`, файл опубліковано.
- `GPT(vocab+1, ...)` із токенайзером меншого vocab → `save_gpt=1`, `reader=0`, файл опубліковано.
- `train_gpt(..., blocks=0)` → `trained=1`, `reader=0`, файл опубліковано: CLI повертав `0`, але отриманий чекпойнт неможливо було завантажити.

Reader мав повний контракт формату (діапазони `vocab/dim/blocks/max_len` + `gpt_config_budget` + збіг vocab із токенайзером), а writer і свіжий запуск — ні. Це суперечить уже наявному правилу для NaN/Inf: writer не має публікувати те, що власний reader відкине.

Рішення — один helper `gpt_config_wire_valid()` для обох сторін:

- reader використовує його замість інлайн-перевірок, тож поведінка на файлах не змінюється;
- `save_gpt` перевіряє конфіг **до** відкриття `.tmp`, а також вимагає `tok.vocab_size == gpt.vocab_size` і `tok.id2word.size() == gpt.vocab_size`;
- свіжий `train_gpt` перевіряє згенерований конфіг одразу після `tok.build()`, до побудови `GPT`. Це закриває і випадок `dim=15`, який раніше кидав необроблений `std::invalid_argument` замість контрольованого `false`.

Тести: `gpt_checkpoint_check` перевіряє відмову writer-а для `blocks=0` і розсинхрону vocab разом із відсутністю файлу й `.tmp`; `gpt_resume_checks` додає відмову свіжого запуску для `blocks=0` і `dim=15`. Мутаційні контролі: вимкнення guard-а writer-а ламає `save rejects blocks=0`; вимкнення guard-а свіжого запуску ламає `gpt_resume_checks` (недосяжний `dim=15` аварійно завершує харнес). Після відновлення точного вихідного файлу — знову `All tests passed.`

Верифікація після фіксу: `build\test.bat` → `All tests passed.`; `build\compile.bat` → `exit=0`; E2E на `build\shade.exe`: train/infer/resume → `0/0/0`, розмір чекпойнта не змінився. Той самий негативний харнес після фіксу дає `saved=0 file=0` для обох writer-кейсів і `trained=0 file=0` для `blocks=0`.

### Ще не зроблено

- `train_config.json` не читається `gpt.cpp` — конфіг зашитий у коді.
- SHDG v3 не має fingerprint датасоту: resume з іншим текстом дозволений навмисно (fine-tuning path). `batch_size` уже серіалізується, але наявність/вміст датасету не перевіряється (P3).
- `rand_calls` має окремі діапазони збереження й replay; явний resume понад 100M відхиляється, старі/довгі чекпойнти лишаються читаними для inference (P3: саме 100M replay перевірено окремим контролем, не щоденним тестом).
- Power-loss durability: `fflush`/`fclose` без `fsync` файлу й каталогу дають атомарну **видимість** за crash процесу, але не гарантію після знеструмлення (POSIX `rename`, Windows `MOVEFILE_WRITE_THROUGH`).
- Формати SHDE generation-файлів і SHDG v2 (GPT) лишаються окремими — уніфікація не потрібна.
- Authenticode-підпис `shade.exe`: потрібен code-signing сертифікат (зараз лише VERSIONINFO-метадані).

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
