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
build\shade.exe                         # train: data/tinystories_valid.txt -> gpt_model.bin
build\shade.exe --load <model.bin> "prompt"
```

`--train` поки **не парситься** — якщо не передано `--load`, програма завжди запускає тренування.
`--load` вимагає шлях до моделі **і** prompt.

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
| `src/utils/serialize.h` | SHDE checkpoint, перевірка `fread`, NaN/Inf |
| `src/utils/tokenizer_bpe.h` | власний BPE tokenizer |

---

## ✅ Що перевірено

### Тести

```
tests/simd_check.cpp    — AVX vs scalar: exp/dot/softmax/matmul/attention/loss/AdamW
tests/gradcheck.cpp     — central finite differences для Dense/LayerNorm/GRU/Attention
tests/tensor_checks.cpp — runtime перевірки навіть при /DNDEBUG
tests/bench.cpp         — scalar vs AVX: matmul/attention/AdamW
```

Компіляція, наприклад:

```
cl /nologo /std:c++17 /O2 /arch:AVX /openmp /EHsc /I src tests\gradcheck.cpp /Fe:build\gradcheck.exe
```

Останній результат `gradcheck` ( tolerance `2e-3` ):

```
Dense        OK  worst rel err 1.18e-05
LayerNorm    OK  worst rel err 2.42e-05
GRU          OK  worst rel err 1.45e-05
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
- Serialization: перевірка кількості параметрів, всіх `fread`, розмірів і NaN/Inf.

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
4. Прогнати: `simd_check`, `gradcheck`, `tensor_checks`, `demos`.
5. Не додавати нові архітектури, доки gradient checking не покриє решту шарів і TransformerBlock повністю.

---

## 🗄 Історичний контекст (не відповідає поточному дереву)

Стара версія цього файлу описувала `src/shade.cpp`, чат у браузері, SFT/DPO, 8-блоковий GPT і checkpoint-файли з `model/*.bin`. Ці файли/команди були видалені або не зібрані в поточному `main`. Історія збережена в git, але орієнтуватися на неї як на активний стан не варто.
