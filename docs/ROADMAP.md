# Research Roadmap

## Completed
- [x] **Week 1** — Spectral compressibility (`NER`, `r90/r95/r99`) & head-rank Spearman agreement across `en`, `es`, `fi`, `qu` on `Qwen3-1.7B`.
- [x] **Week 1b** — Geometric subspace overlap & energy transfer efficiency (`head` vs `joint`).
- [x] **Week 2** — Held-out article-ID splits, out-of-sample noise floor, forward-pass projection, paired bootstrap CIs.
- [x] **Week 3** — Multilingual calibration (`multi` vs `multi_loo`) & `K` vs `V` comparison.
- [x] **Week 4** — Low-budget mixture sweep (`5k / 20k / 150k` tokens, weights `0.00..1.00`) discovering the `5k`-token + `joint` recipe.
- [x] **Week 5** — Position-0 attention-sink control (solved the `K` blow-up & all-layer collapse) + FLORES-200 parallel-corpus control (`flores_to_flores` and `wiki_to_flores`).

## Next Steps

### 1. Week 5b — Deep Dive on `K` & Full-Model `K+V` (`notebooks/06_week5b_k_deepdive.ipynb`, ready to run)
- Measure per-position norms (`t = 0..50`) and exact angular distortion of `K[0]` after Qwen3's `k_norm` RMSNorm.
- Compare 4 position-0 treatments, including **standard full-sequence SVD (`SKIP_FIRST_TOKENS = 0`)** vs explicit `protect_pos0` and 4-sink `protect_pos0_3`.
- Compare `pre-k_norm` (`k_proj`) vs `post-k_norm` (`k_norm`) compression of `K`.
- Map cross-lingual `K` vs `V` bias across all 4 depth quartiles (`0–6`, `7–13`, `14–20`, `21–27`) and evaluate **full-model 4× `K+V` compression** (`all` layers `0–27` and `layers 3–27`) with `protect_pos0 = True` and `mix_5k`.

### 2. Week 6 — Second Model Family (`Llama-3.2-1B` / `Llama-3.2-3B`)
- Compact replication of the winning suite (`protect_pos0 = True`, `K`, `V`, and `K+V`, `head` vs `joint`, `matched / en / multi_loo / multi / mix_5k`) on a model **without QK-Norm** and with a different tokenizer/pretraining mix.
- Isolates how much of the `K[0]` sensitivity is QK-Norm-specific vs universal across decoder-only Transformers.
- Optional: add 1 non-Latin low-resource language from FLORES-200 (e.g., Amharic `amh_Ethi` or Burmese `mya_Mymr`) to separate low-resource effects from script/typology effects.

### 3. Week 7 — Downstream Evaluation Beyond Perplexity
- **SIB-200** (7-class topic classification over FLORES-200, includes `quy_Latn`, `fin_Latn`, `spa_Latn`, `eng_Latn`): fast few-shot log-likelihood accuracy/F1 under compressed KV cache.
- **FLORES-200 translation** (`qu → es` / `qu → en`, 5-shot chrF++): validates autoregressive decoding with compressed KV cache.

### 4. Paper Drafting Checklist
- [ ] Consolidate all CSVs from Kaggle outputs into `results/tables/`.
- [ ] Verify exact definitions against KV-CoRE / Palu / EigenAttention / StreamingLLM baselines.
- [ ] Draft paper sections: Introduction, Method (Gram SVD + `joint` vs `head` + `protect_pos0`), Spectral & Parallel Results, `K` Sink Analysis, Cross-Model Replication, Practical Recommendation (`joint` + `protect_pos0` + `5k` target tokens).
