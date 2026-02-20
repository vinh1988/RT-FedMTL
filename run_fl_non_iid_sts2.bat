@echo off
echo ========================================
echo Federated Learning - Non-IID Distribution
echo MiniLM with Standard FL (3 SST2 Clients)
echo Non-IID Data Distribution (alpha=0.5)
echo ========================================

REM ===== Conda =====
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312

REM ===== Project paths =====
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS
set WORK_DIR=%PROJECT_ROOT%\experiment_new_solution\models\mini-lm\fl-slms-mini-lm-non-iid-sst2

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
echo Starting Federated Learning Server (Non-IID SST2)...
start "FL Non-IID SST2 Server" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode server --config federated_config.yaml --port 8771"

timeout /t 20 >nul

REM ===== SST2 Client 1 =====
echo Starting SST2 Client 1 (with Non-IID)...
start "FL Non-IID SST2 Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2 --port 8771"

timeout /t 3 >nul

REM ===== SST2 Client 2 =====
echo Starting SST2 Client 2 (with Non-IID)...
start "FL Non-IID SST2 Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2 --port 8771"

timeout /t 3 >nul

REM ===== SST2 Client 3 =====
echo Starting SST2 Client 3 (with Non-IID)...
start "FL Non-IID SST2 Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2 --port 8771"

echo.
echo ========================================
echo Federated Learning Non-IID SST2 Started!
echo Configuration:
echo - Total Clients: 3
echo - SST2 Clients: 3 (sst2_client_1, sst2_client_2, sst2_client_3)
echo - Model: MiniLM with Standard FL
echo - Data Distribution: Non-IID (alpha=0.5) on split SST2 dataset
echo - Samples per Client: 22,449 training + 290 validation
echo - Total Dataset: 68,221 (100% utilization, no duplication)
echo - Output: %WORK_DIR%\fl_non_iid_sts2_result
echo ========================================
echo.
echo True federated learning with Non-IID creates optimal data distribution!
echo Each client gets unique 1/3 of SST2 dataset with heterogeneity.
echo.
pause
