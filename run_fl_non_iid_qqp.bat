@echo off
echo ========================================
echo Federated Learning - Non-IID Distribution
echo MiniLM with Standard FL (3 QQP Clients)
echo Non-IID Data Distribution (alpha=0.5)
echo ========================================

REM ===== Conda =====
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312

REM ===== Project paths =====
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS
set WORK_DIR=%PROJECT_ROOT%\experiment_new_solution\models\mini-lm\fl-slms-mini-lm-non-iid-qqp

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
echo Starting Federated Learning Server (Non-IID QQP)...
start "FL Non-IID QQP Server" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode server --config federated_config.yaml --port 8771"

timeout /t 20 >nul

REM ===== QQP Client 1 =====
echo Starting QQP Client 1 (with Non-IID)...
start "FL Non-IID QQP Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_1 --tasks qqp --port 8771"

timeout /t 3 >nul

REM ===== QQP Client 2 =====
echo Starting QQP Client 2 (with Non-IID)...
start "FL Non-IID QQP Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_2 --tasks qqp --port 8771"

timeout /t 3 >nul

REM ===== QQP Client 3 =====
echo Starting QQP Client 3 (with Non-IID)...
start "FL Non-IID QQP Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_3 --tasks qqp --port 8771"

echo.
echo ========================================
echo Federated Learning Non-IID QQP Started!
echo Configuration:
echo - Total Clients: 3
echo - QQP Clients: 3 (qqp_client_1, qqp_client_2, qqp_client_3)
echo - Model: MiniLM with Standard FL
echo - Data Distribution: Non-IID (alpha=0.5) on split QQP dataset
echo - Samples per Client: 107,805 training + ~13,473 validation
echo - Total Dataset: 323,415 (100% utilization, no duplication)
echo - Output: %WORK_DIR%\fl_non_iid_qqp_result
echo ========================================
echo.
echo True federated learning with Non-IID creates optimal data distribution!
echo Each client gets unique 1/3 of QQP dataset with heterogeneity.
echo.
pause
