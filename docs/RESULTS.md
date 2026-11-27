# Quantitative Results Log (Weeks 1–5)

Model: **`Qwen/Qwen3-1.7B`** (`28` layers, `16` query heads, `8` KV heads, `head_dim = 128`, `D_KV = 1024`, late layers = `21–27`).  
Languages: English (`en` / `eng_Latn`), Spanish (`es` / `spa_Latn`), Finnish (`fi` / `fin_Latn`), Ayacucho Quechua (`qu` / `quy_Latn`).  
All paired confidence intervals are 95% bootstrap CIs (`N_BOOT = 2000`) over evaluation windows (`SEQ_LEN = 512`).

---

## Week 1 — Spectral Structure & Head-Level Compressibility (`01_week1_spectra.ipynb`)

Wikipedia `20231101`, `200,000` tokens/language (`390` windows × `512` tokens, `SKIP_FIRST_TOKENS = 1`).

| Lang | Articles for 200k tok | Baseline PPL | K NER | K `r90` | K `r95` | K `r99` | V NER | V `r90` | V `r95` | V `r99` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `en` | 551 | 15.88 | 0.642 | 0.246 | 0.420 | 0.741 | 0.907 | 0.701 | 0.830 | 0.961 |
| `es` | 477 | 13.21 | 0.643 | 0.248 | 0.421 | 0.741 | 0.902 | 0.691 | 0.824 | 0.959 |
| `fi` | 497 | 15.47 | 0.637 | 0.240 | 0.408 | 0.732 | 0.881 | 0.650 | 0.796 | 0.951 |
| `qu` | 964 | 37.15 | 0.611 | 0.207 | 0.368 | 0.705 | 0.863 | 0.617 | 0.771 | 0.944 |

- **Cross-language head-rank agreement (within-layer Spearman):**
  - `K`: `0.958–0.986` (`en-qu`: `0.958`, `en-fi`: `0.960`, `en-es`: `0.986`).
  - `V`: `0.920–0.971` (`en-qu`: `0.929`, `en-fi`: `0.920`, `en-es`: `0.971`).

---

## Week 1b — Subspace Transfer Efficiency (`02_week1b_subspace_transfer.ipynb`)

Mean energy transfer efficiency $\operatorname{tr}(U_A^\top G_B U_A) / \operatorname{tr}(U_B^\top G_B U_B)$ at `r = 0.25·d` across all 28 layers:

| Kind | Source → Target | Per-Head | Joint Layer |
|---|---|---:|---:|
| `K` | `en → es` | 0.981 | 0.986 |
| `K` | `en → fi` | 0.950 | 0.966 |
| `K` | `en → qu` | 0.963 | 0.980 |
| `V` | `en → es` | 0.905 | 0.940 |
| `V` | `en → fi` | 0.802 | 0.881 |
| `V` | `en → qu` | 0.811 | 0.912 |

---

## Week 2 — Held-Out Splits, Noise Floor & Forward-Pass Intervention (`03_week2_intervention_noise_floor.ipynb`)

Article-ID CRC32 split (`150k` calib / `60k` eval tokens per language).  
Uncompressed baseline PPL on eval split: `en = 16.39`, `es = 12.53`, `fi = 15.89`, `qu = 35.53`.

### Held-out energy efficiency in late layers (`21–27`, `r = 0.25`)

| Kind | Scope | Source | `→ en` | `→ es` | `→ fi` | `→ qu` |
|---|---|---|---:|---:|---:|---:|
| `K` | `head` | `en` | **0.998** | 0.964 | 0.908 | 0.953 |
| `K` | `joint` | `en` | **0.996** | 0.971 | 0.933 | 0.971 |
| `V` | `head` | `en` | **0.975** | 0.825 | 0.659 | 0.734 |
| `V` | `joint` | `en` | **0.980** | 0.899 | 0.831 | 0.904 |

*(Bold diagonal = held-out same-language noise floor; own-language diagonal for `es/fi/qu` on `V` is `0.985–0.994`.)*

### Perplexity increase (%) on `V` (`r = 0.25`, `protect_pos0 = False`)

| Layers | Scope | Target | Matched | `en` basis | `oracle` |
|---|---|---|---:|---:|---:|
| `all` (0–27) | `head` | `en` / `qu` | 1783.7% / 552.7% | 1783.7% / 2739.5% | 1447.4% / 539.5% |
| `all` (0–27) | `joint` | `en` / `qu` | 244.6% / 349.7% | 244.6% / 514.1% | 214.8% / 326.8% |
| `late` (21–27) | `head` | `en` | 11.90% | — | 9.72% |
| `late` (21–27) | `head` | `es` | 12.61% | 14.37% | 11.29% |
| `late` (21–27) | `head` | `fi` | 20.70% | 30.81% | 19.64% |
| `late` (21–27) | `head` | `qu` | 10.08% | **43.49%** | 9.79% |
| `late` (21–27) | `joint` | `en` | 5.24% | — | 3.92% |
| `late` (21–27) | `joint` | `es` | 7.66% | 9.67% | 6.70% |
| `late` (21–27) | `joint` | `fi` | 15.37% | 22.12% | 14.54% |
| `late` (21–27) | `joint` | `qu` | 6.79% | **17.10%** | 5.86% |

---

## Week 3 — Multilingual Calibration `multi` vs `multi_loo` (`04_week3_4_multilingual_and_mixtures.ipynb`)

Late-layer `V`, `r = 0.25`, `ΔNLL` vs matched basis (nats/token, 95% bootstrap CI):

| Scope | Target | `en` basis | `multi_loo` (excl. target) | `multi` (incl. target) | `oracle` |
|---|---|---:|---:|---:|---:|
| `head` | `es` | `+0.0156` `[0.0109, 0.0201]` | `+0.0035` `[-0.0015, 0.0085]` | `-0.0060` `[-0.0100, -0.0019]` | `-0.0118` |
| `head` | `fi` | `+0.0805` `[0.0698, 0.0902]` | `+0.0353` `[0.0263, 0.0445]` | `+0.0011` `[-0.0036, 0.0057]` | `-0.0088` |
| `head` | `qu` | **`+0.2650`** `[0.2315, 0.2978]` | **`+0.2589`** `[0.2271, 0.2905]` | `+0.0200` `[0.0151, 0.0248]` | `-0.0027` |
| `joint` | `es` | `+0.0185` `[0.0150, 0.0222]` | `+0.0028` `[0.0000, 0.0056]` | `+0.0019` `[0.0002, 0.0036]` | `-0.0089` |
| `joint` | `fi` | `+0.0569` `[0.0497, 0.0646]` | `+0.0335` `[0.0267, 0.0407]` | `+0.0098` `[0.0075, 0.0121]` | `-0.0072` |
| `joint` | `qu` | **`+0.0921`** `[0.0818, 0.1031]` | **`+0.0793`** `[0.0695, 0.0884]` | **`-0.0008`** `[-0.0029, 0.0012]` | `-0.0088` |

---

## Week 4 — Low-Budget Target-Language Mixture Sweep (`04_week3_4_multilingual_and_mixtures.ipynb`)

Mixing target-language Gram (`5k`, `20k`, or `150k` tokens) at weight `w` with `multi_loo` (`(1 - w)·multi_loo + w·target`), late-layer `V`, `r = 0.25`, `ΔNLL` vs matched:

| Target | Scope | `w = 0.00` (`multi_loo`) | `w = 0.02` (`5k` / `150k`) | `w = 0.05` (`5k` / `150k`) | `w = 0.10` (`5k` / `150k`) | `w = 0.25` (`5k` / `150k`) |
|---|---|---:|---:|---:|---:|---:|
| `fi` | `head` | 0.035 | 0.011 / 0.011 | 0.006 / 0.008 | 0.001 / 0.003 | -0.001 / 0.001 |
| `fi` | `joint` | 0.034 | 0.009 / 0.010 | 0.007 / 0.008 | 0.007 / 0.009 | 0.009 / 0.010 |
| `qu` | `head` | 0.259 | 0.166 / 0.174 | 0.109 / 0.117 | 0.058 / 0.065 | 0.022 / 0.020 |
| `qu` | `joint` | 0.079 | **0.015** / 0.017 | **0.004** / 0.004 | **0.001** / 0.001 | **0.000** / -0.001 |

---

## Week 5 — Position-0 Sink Control & FLORES-200 Parallel Control (`05_week5_pos0_and_flores.ipynb`)

### Part A — Position-0 Control on Wikipedia (`protect_pos0 = False` vs `True`)

#### A2: `K` (late layers 21–27) — PPL increase (%)
| Scope | `r / d` | Target | Source | `protect_pos0 = False` | `protect_pos0 = True` |
|---|---:|---|---|---:|---:|
| `head` | 0.25 | `en` | `matched` | 175.77% | **1.38%** |
| `head` | 0.25 | `fi` | `matched` / `en` | 337.13% / 372.33% | **1.15%** / **2.23%** |
| `head` | 0.25 | `qu` | `matched` / `en` | 158.65% / 186.76% | **1.99%** / **4.98%** |
| `head` | 0.75 | `en` / `fi` / `qu` | `matched` | 68.83% / 119.62% / 57.28% | **-0.15%** / **0.07%** / **0.25%** |
| `joint` | 0.25 | `en` | `matched` | 47.73% | **0.18%** |
| `joint` | 0.25 | `fi` | `matched` / `en` | 78.54% / 80.97% | **0.41%** / **0.44%** |
| `joint` | 0.25 | `qu` | `matched` / `en` | 37.37% / 43.06% | **0.69%** / **2.18%** |

#### A3: `V` (all layers 0–27, `joint`, `r = 0.25`) — PPL increase (%)
| Target | Source | `protect_pos0 = False` | `protect_pos0 = True` |
|---|---|---:|---:|
| `en` | `matched` | 244.62% | **34.13%** |
| `qu` | `matched` | 349.73% | **7.86%** |
| `qu` | `en` | 513.96% | **70.00%** |

#### A1: `V` (late layers 21–27, `r = 0.25`) — `protect_pos0 = True`
| Scope | Target | Matched PPL (%) | `en` PPL (%) | `ΔNLL` (`en` vs matched) | 95% CI | `ΔNLL` (`multi` vs matched) |
|---|---|---:|---:|---:|---|---:|
| `head` | `fi` | +6.95% | +18.53% | **+0.1028** | `[0.0921, 0.1129]` | +0.0026 (n.s.) |
| `head` | `qu` | +3.74% | +35.46% | **+0.2668** | `[0.2336, 0.2999]` | +0.0216 |
| `joint` | `fi` | +0.49% | +8.06% | **+0.0726** | `[0.0649, 0.0803]` | +0.0017 (n.s.) |
| `joint` | `qu` | -0.89% | +8.00% | **+0.0859** | `[0.0761, 0.0952]` | -0.0045 |

---

### Part B — Parallel-Corpus Control on FLORES-200 `devtest` (`protect_pos0 = True`, `V`, late layers, `r = 0.25`)

Tokenizer fertility (tokens/sentence on identical text): `en = 26.81`, `es = 40.81`, `fi = 52.92`, `qu = 55.21`.  
Uncompressed FLORES `devtest` baseline PPL: `en = 22.93`, `es = 14.89`, `fi = 21.48`, `qu = 168.61`.  
Equalized FLORES `dev` calibration budget: `53` windows (`27,136` tokens/lang).

#### `flores_to_flores` (strictly parallel calibration & evaluation) — `ΔNLL` vs matched (PPL %)
| Scope | Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` | `oracle` |
|---|---|---:|---:|---:|---:|---:|---:|
| `head` | `es` | +2.42% | `+0.0490` (+7.6%) | `+0.0119` (+3.7%) | `+0.0077` (+3.2%) | `+0.0061` (+3.0%) | `-0.0055` |
| `head` | `fi` | +3.85% | `+0.1407` (+19.5%) | `+0.0747` (+11.9%) | `+0.0123` (+5.1%) | `+0.0060` (+4.5%) | `-0.0027` |
| `head` | `qu` | -3.98% | **`+0.6007`** (+75.1%) | **`+0.5562`** (+67.5%) | `+0.0366` (-0.4%) | `-0.0054` (-4.5%) | `+0.0003` |
| `joint` | `es` | -0.54% | `+0.0511` (+4.7%) | `+0.0164` (+1.1%) | `+0.0051` (-0.0%) | `+0.0033` (-0.2%) | `-0.0037` |
| `joint` | `fi` | +0.22% | `+0.1068` (+11.5%) | `+0.0652` (+7.0%) | **`-0.0003`** (+0.2%) | `+0.0001` (+0.2%) | `-0.0033` |
| `joint` | `qu` | -2.17% | **`+0.1468`** (+13.3%) | **`+0.1493`** (+13.6%) | **`-0.0296`** (-5.0%) | **`-0.0256`** (-4.6%) | `+0.0020` |

#### `wiki_to_flores` (Wikipedia bases → parallel FLORES `devtest`, cross-domain transfer) — `ΔNLL` vs matched (PPL %)
| Scope | Target | Matched PPL | `en` basis | `multi_loo` | `mix_5k` (`w=0.10`) | `multi` | `oracle` |
|---|---|---:|---:|---:|---:|---:|---:|
| `head` | `es` | +4.31% | `+0.0227` (+6.7%) | `+0.0097` (+5.3%) | `+0.0062` (+5.0%) | `+0.0028` (+4.6%) | `-0.0238` |
| `head` | `fi` | +6.08% | `+0.0985` (+17.1%) | `+0.0659` (+13.3%) | **`+0.0004`** (+6.1%) | `-0.0002` (+6.1%) | `-0.0240` |
| `head` | `qu` | -2.43% | **`+0.5183`** (+63.8%) | **`+0.5274`** (+65.3%) | `+0.0952` (+7.3%) | `+0.0209` (-0.4%) | `-0.0158` |
| `joint` | `es` | +1.06% | `+0.0343` (+4.6%) | `+0.0189` (+3.0%) | `+0.0094` (+2.0%) | `+0.0058` (+1.7%) | `-0.0196` |
| `joint` | `fi` | +1.42% | `+0.0874` (+10.7%) | `+0.0744` (+9.3%) | `+0.0106` (+2.5%) | `+0.0055` (+2.0%) | `-0.0152` |
| `joint` | `qu` | -1.58% | **`+0.1295`** (+12.0%) | **`+0.1199`** (+11.0%) | **`-0.0086`** (-2.4%) | **`-0.0111`** (-2.7%) | `-0.0039` |
