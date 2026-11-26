# Offline Quantization Drift Case Study (FP16 -> Q8 -> Q4)

This walkthrough demonstrates an end-to-end, reproducible offline migration check comparing a reference baseline (`ref-fp16`) against two quantized candidates (`cand-q8` and `cand-q4`) across two traffic routes (`billing` extraction and `support` classification).

> **Scope & Honesty Note:** The fixtures in [`examples/case-study/`](../examples/case-study/) are **sanitized synthetic reference captures** crafted to reproduce real-world quantization failure modes deterministically in CI without external API keys or model weights. They do not represent benchmark claims about any specific third-party model vendor or hardware setup.

## Scenario

| Artifact | Role | Key Properties |
| :--- | :--- | :--- |
| `prompts.json` | 6-prompt suite | 3 `billing` extraction prompts + 3 `support` classification prompts, strict schemas (`additionalProperties: false`) |
| `drift.toml` | Evaluation policy | Allows `$.confidence` jitter (`tolerance_abs = 0.03`) and `$.tags` array reordering (`unordered_array = true`) |
| `baseline-fp16.jsonl` | Reference baseline | Complete `latency_ms` & `tokens_generated`; 100% schema-valid |
| `candidate-q8.jsonl` | Safe quantization | ~36% faster P95 latency, minor confidence jitter (`<= 0.01`) and tag reordering tolerated by `drift.toml` -> **PASS** |
| `candidate-q4.jsonl` | Regressed quantization | ~60% faster latency, **but** drops tax from invoice totals (`bill-01`, `bill-02`), flattens support `priority` from `high` to `medium`, and adds an undeclared key on `sup-03` -> **FAIL** |

## Step-by-Step Reproducible Commands

```bash
# 1. Pre-flight check: environment, policy, strict schemas, secret scan, and ID alignment
llm-drift-bench doctor \
  --config examples/case-study/drift.toml \
  -s examples/case-study/prompts.json \
  -b examples/case-study/baseline-fp16.jsonl \
  -c examples/case-study/candidate-q8.jsonl

llm-drift-bench lint-suite -s examples/case-study/prompts.json --strict

# 2. Rank Q8 vs Q4 side-by-side against the FP16 baseline
llm-drift-bench compare-many \
  -b examples/case-study/baseline-fp16.jsonl \
  -s examples/case-study/prompts.json \
  --config examples/case-study/drift.toml \
  -c q8 examples/case-study/candidate-q8.jsonl \
  -c q4 examples/case-study/candidate-q4.jsonl \
  --against q8 --no-fail-on-drift

# 3. Inspect segmented distribution blast radius per route on Q4
llm-drift-bench distribution \
  -s examples/case-study/prompts.json \
  -b examples/case-study/baseline-fp16.jsonl \
  -c examples/case-study/candidate-q4.jsonl \
  --group-by metadata:route --shift-threshold 0.15

# 4. Run unified offline audit (drift + distributions + quality + logprobs + token volume)
llm-drift-bench audit \
  -s examples/case-study/prompts.json \
  -b examples/case-study/baseline-fp16.jsonl \
  -c examples/case-study/candidate-q8.jsonl \
  --config examples/case-study/drift.toml \
  --group-by metadata:route --fail-on-widespread

# 5. Compare token volume (explicit tokens_generated, never fabricated)
llm-drift-bench cost \
  -b examples/case-study/baseline-fp16.jsonl \
  -c examples/case-study/candidate-q4.jsonl
```

## Expected Findings

1. **`cand-q8` passes** every check: `drift.toml` absorbs harmless `$.confidence` jitter (`0.01 <= 0.03`) and unordered `$.tags`, while P95 latency improves from `215 ms` to `140 ms`.
2. **`cand-q4` fails** across multiple independent signals:
   - **Schema validation**: `sup-03` violates `additionalProperties: false` (`extra_note`).
   - **Numeric drift**: `bill-01` and `bill-02` drop tax from `$.total` (`110 -> 100`, `275 -> 250`).
   - **Segmented distribution drift**: `support` route shifts `$.priority` from 100% `high` to 100% `medium` (`TVD = 1.0000`) and shifts `$.confidence` median by `-0.30`.
3. Every command above is verified automatically by [`tests/test_case_study.py`](../tests/test_case_study.py).
