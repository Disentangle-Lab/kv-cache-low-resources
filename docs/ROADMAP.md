# Research Roadmap

## Completed
- [x] **Week 1** — Spectral compressibility (`NER`, `r90/r95/r99`) & head-rank Spearman agreement across `en`, `es`, `fi`, `qu` on `Qwen3-1.7B`.
- [x] **Week 1b** — Geometric subspace overlap & energy transfer efficiency (`head` vs `joint`).
- [x] **Week 2** — Held-out article-ID splits, out-of-sample noise floor, forward-pass projection, paired bootstrap CIs.
- [x] **Week 3** — Multilingual calibration (`multi` vs `multi_loo`) & `K` vs `V` comparison.
- [x] **Week 4** — Low-budget mixture sweep (`5k / 20k / 150k` tokens, weights `0.00..1.00`) discovering the `5k`-token + `joint` recipe.
- [x] **Week 5** — Position-0 attention-sink control (solved the `K` blow-up & all-layer collapse) + FLORES-200 parallel-corpus control (`flores_to_flores` and `wiki_to_flores`).
- [x] **Week 5b** — Deep dive on `K`: proved standard full-sequence SVD still breaks `K[0]`, quantified `k_norm` angular distortion (`cos_sim 0.689` on pos 0), showed `post-k_norm` cuts blow-up >50%, proved `K` is language-agnostic across all depth quartiles while `V` carries all cross-lingual bias, and demonstrated 4× full-model `K+V` (`layers 3–27`) with `joint` + `protect_pos0` + `mix_5k` (`qu: +0.51%`, `fi: +7.27%`, `es: +14.26%` vs `+428%` English-calibrated).

## Next Steps

### 1. Week 6 — Second Model Family (`Llama-3.2-1B` / `Llama-3.2-3B`)
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
