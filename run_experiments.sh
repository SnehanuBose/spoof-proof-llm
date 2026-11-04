#!/usr/bin/env bash
# ==============================================================================
# [EXPERIMENT RUNNER] Offline SLM Prompt Spoofing and Injection Defense
# Author: Snehanu Bose, Final Year B.Tech CST
# Target Hardware: Acer Nitro V15 (RTX 5050 8GB VRAM / 24GB DDR4 Host RAM)
# Python Runtime Target: Python 3.11.7
# ==============================================================================

set -e

echo "=============================================================================="
echo "[EXPERIMENT RUNNER] Offline SLM Prompt Spoofing and Injection Defense"
echo "Author: Snehanu Bose, Final Year B.Tech CST"
echo "Target Hardware: Acer Nitro V15 (RTX 5050 8GB VRAM / 24GB DDR4 Host RAM)"
echo "Python Runtime Target: Python 3.11.7"
echo "=============================================================================="

# Step 1: Verify Python Version
echo "[*] Step 1: Verifying active Python version..."
PYTHON_BIN=$(command -v python3 || command -v python)
DETECTED_VER=$($PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
echo "[INFO] Detected Python Version: ${DETECTED_VER}"

if [ "${DETECTED_VER}" != "3.11.7" ]; then
    echo "[WARNING] Target is Python 3.11.7, but detected ${DETECTED_VER}."
    $PYTHON_BIN -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'" || {
        echo "[ERROR] Incompatible Python version."
        exit 1
    }
fi
echo "[OK] Python version verified."

# Step 2: Install dependencies
echo ""
if [ "$1" == "--no-pip" ] || [ "${SKIP_PIP}" == "1" ]; then
    echo "[*] Step 2: Skipping dependency install (--no-pip or SKIP_PIP=1)."
else
    echo "[*] Step 2: Verifying requirements..."
    $PYTHON_BIN -m pip install -r requirements.txt --quiet || echo "[NOTICE] Offline or partial pip install. Continuing with existing modules."
fi
echo "[OK] Dependencies check completed."

# Step 3: Check Ollama and pull base models if server is running
echo ""
echo "[*] Step 3: Checking Ollama local serving engine..."
if command -v ollama >/dev/null 2>&1; then
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "[INFO] Ollama running. Pulling base models..."
        ollama pull qwen2.5:3b || true
        ollama pull llama3.2:3b || true
        ollama pull phi3.5:latest || true
        ollama pull gemma2:2b || true
    else
        echo "[NOTICE] Ollama server not active at http://localhost:11434."
    fi
else
    echo "[NOTICE] Ollama not found in PATH. Using verified dry-run mode."
fi

# Step 4: Run 4-bit NF4 QLoRA DPO Alignment
echo ""
echo "[*] Step 4: Running QLoRA SecAlign DPO Alignment across all 4 models..."
$PYTHON_BIN multi_model_hybrid_defense.py --mode train --model all --output-dir ./artifacts

# Step 5: Export Modelfiles
echo ""
echo "[*] Step 5: Exporting Ollama Modelfiles..."
$PYTHON_BIN multi_model_hybrid_defense.py --mode export-modelfile --output-dir ./artifacts

# Step 6: Benchmark Evaluation
echo ""
echo "[*] Step 6: Running comparative Before vs. After evaluation benchmark..."
$PYTHON_BIN multi_model_hybrid_defense.py --mode eval --backend dry-run --output-json benchmark_results.json

echo ""
echo "=============================================================================="
echo "[SUCCESS] Experiment pipeline completed successfully!"
echo "Results saved to: benchmark_results.json"
echo "=============================================================================="
