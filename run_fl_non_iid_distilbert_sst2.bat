@echo off
echo ========================================
echo Federated Learning - Non-IID Distribution
echo DistilBERT with Standard FL (3 SST2 Clients)
echo Non-IID Data Distribution (alpha=0.5)
echo ========================================

REM ===== Conda =====
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312

REM ===== Project paths =====
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS
set WORK_DIR=%PROJECT_ROOT%\experiment_new_solution\models\distil-bert\fl-slms-mini-lm-non-iid-sst2

set PYTHONPATH=%PROJECT_ROOT%

echo Project Root: %PROJECT_ROOT%
echo Work Directory: %WORK_DIR%
echo Conda Environment: %CONDA_ENV%

REM Check if conda environment exists
echo Checking conda environment...
%CONDA_ROOT%\condabin\conda.bat info --envs | findstr %CONDA_ENV% >nul
if errorlevel 1 (
    echo ERROR: Conda environment '%CONDA_ENV%' not found!
    echo Please create it first:
    echo conda create -n %CONDA_ENV% python=3.12 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    pause
    exit /b 1
)

echo Environment found: %CONDA_ENV%
echo.

REM ===== Server =====
echo Starting Federated Learning Server for SST2...
start "FL SST2 Server" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode server --config federated_config.yaml"

timeout /t 20 >nul

REM ===== SST2 Client 1 =====
echo Starting SST2 Client 1...
start "FL SST2 Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2 --config federated_config.yaml"

timeout /t 3 >nul

REM ===== SST2 Client 2 =====
echo Starting SST2 Client 2...
start "FL SST2 Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2 --config federated_config.yaml"

timeout /t 3 >nul

REM ===== SST2 Client 3 =====
echo Starting SST2 Client 3...
start "FL SST2 Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2 --config federated_config.yaml"

timeout /t 3 >nul

echo.
echo ========================================
echo All SST2 clients started successfully!
echo ========================================
echo.
echo Server: Port 8771
echo Clients: 3 total (SST2 sentiment analysis)
echo Model: distilbert-base-uncased (DistilBERT, ~66M params)
echo Distribution: Non-IID (alpha=0.5)
echo Dataset: SST2 (66,477 train + 872 val samples)
echo.
echo Each client gets ~22,159 training samples with heterogeneous sentiment distributions
echo Validation: 872 samples per client (full validation set)
echo.
echo Check individual client windows for progress
echo Results will be saved to: fl_non_iid_sts2_result
echo.
pause
