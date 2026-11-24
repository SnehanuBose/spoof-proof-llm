import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone

import requests

OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODELS = {
    "qwen": "qwen2.5:3b",
    "llama": "llama3.2:3b",
    "phi": "phi3.5:latest",
    "gemma": "gemma2:2b",
}
FIELDS = [
    "timestamp_utc", "model_key", "model_tag", "test_id", "attack_category",
    "repetition", "expected_exploit_marker", "marker_found",
    "latency_ms", "prompt", "raw_response", "manual_outcome", "review_notes"
]

def call_model(model_tag, prompt, timeout):
    # No system message, sanitizer, nonce spotlighting, or fine-tuned adapter:
    # this is the repository's raw/no-defense baseline path.
    payload = {
        "model": model_tag,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 128}
    }
    start = time.perf_counter()
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=timeout
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()
    data = response.json()
    return data.get("response", ""), elapsed_ms

def main():
    global OLLAMA_URL
    parser = argparse.ArgumentParser(
        description="Run the repository's original benchmark cases against raw Ollama models."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dataset", default="dataset/repository_benchmark.json")
    parser.add_argument("--out", default="results")
    parser.add_argument("--base-url", default=OLLAMA_URL)
    args = parser.parse_args()

    OLLAMA_URL = args.base_url.rstrip("/")

    unknown = [m for m in args.models if m not in DEFAULT_MODELS and m not in DEFAULT_MODELS.values()]
    if unknown:
        parser.error(f"Unknown model key/tag(s): {unknown}. Choose keys {list(DEFAULT_MODELS)} or exact tags.")



    
    chosen = []

    for model_arg in args.models:
        if model_arg in DEFAULT_MODELS:
            chosen.append((model_arg, DEFAULT_MODELS[model_arg]))
        else:
            model_key = next(
                key for key, tag in DEFAULT_MODELS.items()
                if tag == model_arg
            )
         
            chosen.append((model_key, model_arg))

    os.makedirs(args.out, exist_ok=True)
    with open(args.dataset, encoding="utf-8") as f:
        tests = json.load(f)

    # Fail fast if Ollama is unreachable. Never silently substitute simulated results.
    tags_response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    tags_response.raise_for_status()
    installed = {m.get("name") for m in tags_response.json().get("models", [])}
    for key, tag in chosen:
        if tag not in installed:
            raise SystemExit(
                f"Model '{tag}' ({key}) is not installed in Ollama. "
                f"Run: ollama pull {tag}"
            )

    csv_path = os.path.join(args.out, "raw_baseline_results.csv")
    jsonl_path = os.path.join(args.out, "raw_baseline_responses.jsonl")
    total = len(chosen) * len(tests) * args.repetitions
    done = 0

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as cf, \
         open(jsonl_path, "w", encoding="utf-8") as jf:
        writer = csv.DictWriter(cf, fieldnames=FIELDS)
        writer.writeheader()

        for model_key, model_tag in chosen:
            print(f"\n=== {model_key}: {model_tag} ===")
            for test in tests:
                prompt = (
                    f"Instruction: {test['task_instruction']}\n"
                    f"Context: {test['untrusted_context']}"
                )
                for rep in range(1, args.repetitions + 1):
                    done += 1
                    try:
                        answer, latency = call_model(model_tag, prompt, args.timeout)
                        marker = test.get("expected_exploit_marker")
                        found = bool(marker and marker.casefold() in answer.casefold())
                        row = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model_key": model_key,
                            "model_tag": model_tag,
                            "test_id": test["id"],
                            "attack_category": test["attack_category"],
                            "repetition": rep,
                            "expected_exploit_marker": marker or "",
                            "marker_found": int(found),
                            "latency_ms": round(latency, 2),
                            "prompt": prompt,
                            "raw_response": answer,
                            "manual_outcome": "",
                            "review_notes": ""
                        }
                        writer.writerow(row)
                        cf.flush()
                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jf.flush()
                        print(f"[{done}/{total}] {test['id']} rep {rep}: "
                              f"{'MARKER FOUND' if found else 'marker absent'} | {latency:.0f} ms")
                    except requests.RequestException as exc:
                        print(f"[{done}/{total}] ERROR {model_tag}/{test['id']}: {exc}")
                        # Log failed calls so they are visible, not counted as attack failures.
                        row = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "model_key": model_key, "model_tag": model_tag,
                            "test_id": test["id"], "attack_category": test["attack_category"],
                            "repetition": rep,
                            "expected_exploit_marker": test.get("expected_exploit_marker") or "",
                            "marker_found": "", "latency_ms": "",
                            "prompt": prompt, "raw_response": "",
                            "manual_outcome": "REQUEST_ERROR", "review_notes": str(exc)
                        }
                        writer.writerow(row)
                        cf.flush()
                        jf.write(json.dumps(row, ensure_ascii=False) + "\n")
                        jf.flush()

    print(f"\nFinished. CSV: {csv_path}\nRaw JSONL: {jsonl_path}")
    print("No simulated fallback was used. Review each response before reporting ASR.")

if __name__ == "__main__":
    main()
