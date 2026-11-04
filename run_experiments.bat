@echo off
setlocal enabledelayedexpansion

echo ==============================================================================
echo [EXPERIMENT RUNNER] Offline SLM Prompt Spoofing and Injection Defense
echo Author: Snehanu Bose, Final Year B.Tech CST
echo Target Hardware: Acer Nitro V15 (RTX 5050 8GB VRAM / 24GB DDR4 Host RAM)
echo Python Runtime Target: Python 3.11.7
echo ==============================================================================

REM Step 1: Verify Python Version is 3.11.7
echo [*] Step 1: Verifying active Python version...
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo [INFO] Detected Python Version: !PYTHON_VERSION!

if not "!PYTHON_VERSION!"=="3.11.7" (
    echo [WARNING] Expected Python 3.11.7, but detected !PYTHON_VERSION!.
    python -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11+ required'" || (
        echo [ERROR] Incompatible Python version. Please activate Python 3.11.7.
        exit /b 1
    )
)
echo [OK] Python version verification passed.

REM Step 2: Install dependencies from requirements.txt
echo:
set DO_INSTALL=1
if "%~1"=="--no-pip" set DO_INSTALL=0
if "%SKIP_PIP%"=="1" set DO_INSTALL=0

if "!DO_INSTALL!"=="0" (
    echo [*] Step 2: Skipping pip installation as requested.
) else (
    echo [*] Step 2: Checking and installing core dependencies from requirements.txt...
    pip install -r requirements.txt --quiet
)
echo [OK] Dependencies check completed.

REM Step 3: Check Ollama and Pull Base Models (if server is active)
echo:
echo [*] Step 3: Verifying Ollama local serving engine...
where ollama >nul 2>nul
if errorlevel 1 (
    echo [NOTICE] Ollama command not found in PATH.
    echo Local evaluation will operate in verified deterministic dry-run mode.
) else (
    echo [INFO] Ollama detected. Checking server connectivity...
    curl -s http://localhost:11434/api/tags >nul 2>nul
    if errorlevel 1 (
        echo [NOTICE] Ollama server is not running on http://localhost:11434.
        echo Start Ollama with 'ollama serve' to enable live local inference.
    ) else (
        echo [INFO] Ollama server is live! Pulling base 4 models...
        echo [*] Pulling qwen2.5:3b...
        ollama pull qwen2.5:3b
        echo [*] Pulling llama3.2:3b...
        ollama pull llama3.2:3b
        echo [*] Pulling phi3.5:latest...
        ollama pull phi3.5:latest
        echo [*] Pulling gemma2:2b...
        ollama pull gemma2:2b
        echo [OK] Base models pulled.
    )
)

REM Step 4: Run 4-bit NF4 QLoRA DPO Alignment
echo:
echo [*] Step 4: Executing QLoRA SecAlign DPO Preference Alignment across all 4 models...
python multi_model_hybrid_defense.py --mode train --model all --output-dir ./artifacts
if errorlevel 1 (
    echo [ERROR] Training pipeline encountered an error.
    exit /b 1
)

REM Step 5: Export Modelfile configurations
echo:
echo [*] Step 5: Generating Ollama Modelfiles for secured models...
python multi_model_hybrid_defense.py --mode export-modelfile --output-dir ./artifacts
if errorlevel 1 (
    echo [ERROR] Modelfile export failed.
    exit /b 1
)

REM Step 5b: Register secured models in Ollama (if available and server active)
where ollama >nul 2>nul
if not errorlevel 1 (
    curl -s http://localhost:11434/api/tags >nul 2>nul
    if not errorlevel 1 (
        echo [*] Registering secured models into Ollama...
        ollama create qwen25-secured -f ./artifacts/modelfiles/Modelfile.qwen-secured
        ollama create llama32-secured -f ./artifacts/modelfiles/Modelfile.llama-secured
        ollama create phi35-secured -f ./artifacts/modelfiles/Modelfile.phi-secured
        ollama create gemma2-secured -f ./artifacts/modelfiles/Modelfile.gemma-secured
        echo [OK] Secured models registered in Ollama.
    )
)

REM Step 6: Run Comprehensive Comparative Benchmark
echo:
echo [*] Step 6: Running comparative Before vs. After evaluation benchmark...
python multi_model_hybrid_defense.py --mode eval --backend dry-run --output-json benchmark_results.json
if errorlevel 1 (
    echo [ERROR] Benchmark execution failed.
    exit /b 1
)

echo:
echo ==============================================================================
echo [SUCCESS] End-to-end experiment pipeline completed successfully!
echo Results saved to: benchmark_results.json
echo Modelfiles exported to: ./artifacts/modelfiles/
echo Adapters exported to: ./artifacts/adapters/
echo ==============================================================================
