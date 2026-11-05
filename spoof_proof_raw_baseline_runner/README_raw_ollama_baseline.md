# Raw Ollama Baseline Evaluation

A standalone, no-defense evaluation runner for the benchmark cases used by
`spoof-proof-llm`. It sends the benchmark prompts to locally installed Ollama
models and records their actual responses. It does **not** use the repository's
simulated dry-run responses.

## Purpose

This experiment establishes a raw/no-defense baseline for four locally hosted
models before evaluating any defense mechanisms:

| Model | Ollama tag |
|---|---|
| Qwen 2.5 3B | `qwen2.5:3b` |
| Llama 3.2 3B | `llama3.2:3b` |
| Phi-3.5 Mini | `phi3.5:latest` |
| Gemma 2 2B | `gemma2:2b` |

## Benchmark

The benchmark file, `dataset/repository_benchmark.json`, contains **11 cases**:

| Category | Test IDs | Cases |
|---|---|---:|
| Naive injection | `NAIVE-01`, `NAIVE-02` | 2 |
| Ignore override | `OVERRIDE-01`, `OVERRIDE-02` | 2 |
| Fake completion / delimiter tokens | `DELIMITER-01` to `DELIMITER-04` | 4 |
| Delimiter forgery | `FORGERY-01` | 1 |
| Clean utility controls | `CLEAN-01`, `CLEAN-02` | 2 |
| **Total** | | **11** |

This is 9 attack cases plus 2 clean utility controls.

## Experimental protocol

- Prompts use the format `Instruction: ...\nContext: ...`, matching the raw
  prompt construction used for the benchmark.
- Requests are sent to the local Ollama `/api/generate` endpoint.
- Generation options: `temperature=0.1`, `num_predict=128`, `stream=false`.
- No system prompt, sanitizer, nonce spotlighting, PromptGuard classifier,
  secured adapter, or fine-tuning is applied by this runner.
- The runner checks that Ollama is reachable and that each requested model tag
  is installed.
- Request failures are logged as `REQUEST_ERROR`; the runner does not silently
  replace live model output with synthetic output.
- Each response is written to CSV and JSONL as the run progresses.

## Initial experimental findings

The following results are from one run of each of the four models against all
11 benchmark cases (44 model-test executions: 36 attack evaluations and 8
clean-control evaluations).

### Attack marker hit rate

A marker hit means the expected exploit-marker string was found in the model's
raw response. It is a **first-pass lexical metric**, not a semantic judgement
that the model obeyed the malicious instruction.

| Model | Marker hits | Attack cases | Marker hit rate |
|---|---:|---:|---:|
| Gemma 2 2B | 3 | 9 | 33.33% |
| Llama 3.2 3B | 4 | 9 | 44.44% |
| Phi-3.5 Mini | 5 | 9 | 55.56% |
| Qwen 2.5 3B | 6 | 9 | 66.67% |
| **Overall** | **18** | **36** | **50.00%** |

Lower marker-hit rate is preferable for this preliminary metric. In this run,
Gemma had the lowest marker-hit rate and Qwen had the highest. This is not a
definitive ranking of prompt-injection resistance: a response can mention a
marker while rejecting it, and an attack can succeed without reproducing the
exact marker.

### Marker hit rate by attack category

Percentages are marker hits divided by the number of cases in that category
for each model.

| Attack category | Gemma 2 2B | Llama 3.2 3B | Phi-3.5 Mini | Qwen 2.5 3B |
|---|---:|---:|---:|---:|
| Naive injection (2 cases) | 50% | 50% | 100% | 50% |
| Ignore override (2 cases) | 50% | 50% | 50% | 50% |
| Fake completion (4 cases) | 25% | 50% | 50% | 75% |
| Delimiter forgery (1 case) | 0% | 0% | 0% | 100% |

The delimiter-forgery category contains only one test case, so its percentages
are particularly sensitive to that single response.

### Clean utility controls

All four models produced responses that completed both clean utility tasks in
this run: `CLEAN-01` (summarize the mitochondria/ATP fact) and `CLEAN-02`
(extract the primary programming language, Rust). This is an initial
observation from only two controls per model, not evidence of general utility
preservation.

## Interpretation and limitations

1. The experiment uses one repetition per test and model. Results may vary
   across repeated generations.
2. Exact marker matching can produce false positives when a model quotes,
   discusses, or rejects an injected marker.
3. Marker absence does not prove resistance: a model may follow an injected
   instruction without reproducing the expected marker.
4. The semantic attack success rate (ASR) has **not yet been finalized**.
   Responses require manual review using clearly defined labels such as
   `SUCCESS`, `PARTIAL`, `RESISTED`, `TASK_FAILED`, or `AMBIGUOUS`.
5. The clean-control sample is small (two tasks per model).
6. Latency depends on local hardware, model state, Ollama version, and runtime
   conditions. Record the environment before comparing latency across
   machines.

For research reporting, keep marker-hit rate and manually validated semantic
ASR as separate metrics. Do not report marker-hit rate as semantic ASR.

## Repository evaluation implementation note

Inspection of the source benchmark implementation found that the repository's
`dry-run` evaluation path uses simulated responses rather than querying the
local models. The Ollama path in the original implementation also has a
simulation fallback on request exceptions. This standalone runner was created
to avoid that ambiguity: it queries Ollama directly, checks installed model
tags, and records request errors rather than substituting simulated responses.

This is an implementation observation about the inspected repository version;
re-check the source if the upstream implementation changes.

## Setup

Requirements: Python 3, Ollama, and the Python packages `requests` and
`pandas`.

```powershell
py -m pip install requests pandas
ollama list
```

Pull any missing models:

```powershell
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull phi3.5:latest
ollama pull gemma2:2b
```

## Run the baseline

Run these commands from this project directory:

```powershell
python src/run_raw_baseline.py --models qwen llama phi gemma --repetitions 1
python src/summarize_baseline.py
```

For a single-model pilot:

```powershell
python src/run_raw_baseline.py --models qwen --repetitions 1
```

For repeated runs:

```powershell
python src/run_raw_baseline.py --models qwen llama phi gemma --repetitions 3
```

The runner writes:

```text
results/
├── raw_baseline_results.csv
└── raw_baseline_responses.jsonl
```

The CSV includes the prompt, raw response, marker result, latency, and empty
manual-review fields. The JSONL file preserves each response as a separate
record. Preserve each run's outputs when collecting repetitions.

## Reproducibility checklist

When publishing or sharing results, record:

- Date and time of the run
- Operating system and hardware
- Python version
- Ollama version
- Exact output of `ollama list`, including model tags and digests
- Model quantization where available
- Runner version and benchmark dataset revision
- Temperature, token limit, repetition count, and any prompt changes

## Scope

This folder contains the standalone raw/no-defense baseline runner and its
benchmark transcription. It does not implement or evaluate the repository's
defense mechanisms. Defense evaluation should use a separately documented
protocol and preserve comparability with this baseline.
