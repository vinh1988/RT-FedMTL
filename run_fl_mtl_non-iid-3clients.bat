@echo off
echo ========================================
echo Federated Learning - Multi-Task Learning (MTL)
echo MiniLM with 3 Clients (SST2 + QQP + STSB)
echo Non-IID Data Distribution (alpha=0.5)
echo ========================================

REM ===== Conda =====
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312

REM ===== Project paths =====
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS
set WORK_DIR=%PROJECT_ROOT%\experiment_new_solution\models\mini-lm\fl-mtl-slms-mini-lm-non-iid-sst2-qqp-sst2-3client-each

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
echo Starting FL-MTL Server (3 Clients: SST2 + QQP + STSB)...
start "FL-MTL Server" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode server --config federated_config.yaml --port 8772"

timeout /t 20 >nul

REM ===== SST2 Clients =====
echo Starting SST2 Client 1 (with Non-IID)...
start "FL-MTL SST2 Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2 --port 8772"

timeout /t 3 >nul

echo Starting SST2 Client 2 (with Non-IID)...
start "FL-MTL SST2 Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2 --port 8772"

timeout /t 3 >nul

echo Starting SST2 Client 3 (with Non-IID)...
start "FL-MTL SST2 Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2 --port 8772"

timeout /t 3 >nul

REM ===== QQP Clients =====
echo Starting QQP Client 1 (with Non-IID)...
start "FL-MTL QQP Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_1 --tasks qqp --port 8772"

timeout /t 3 >nul

echo Starting QQP Client 2 (with Non-IID)...
start "FL-MTL QQP Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_2 --tasks qqp --port 8772"

timeout /t 3 >nul

echo Starting QQP Client 3 (with Non-IID)...
start "FL-MTL QQP Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_3 --tasks qqp --port 8772"

timeout /t 3 >nul

REM ===== STSB Clients =====
echo Starting STSB Client 1 (with Non-IID)...
start "FL-MTL STSB Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_1 --tasks stsb --port 8772"

timeout /t 3 >nul

echo Starting STSB Client 2 (with Non-IID)...
start "FL-MTL STSB Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_2 --tasks stsb --port 8772"

timeout /t 3 >nul

echo Starting STSB Client 3 (with Non-IID)...
start "FL-MTL STSB Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_3 --tasks stsb --port 8772"

echo.
echo ========================================
echo FL-MTL Training Started!
echo Configuration:
echo - Total Clients: 9
echo - SST2 Clients: 3 (sst2_client_1, sst2_client_2, sst2_client_3)
echo - QQP Clients: 3 (qqp_client_1, qqp_client_2, qqp_client_3)
echo - STSB Clients: 3 (stsb_client_1, stsb_client_2, stsb_client_3)
echo - Model: MiniLM with Multi-Task Learning
echo - Data Distribution: Non-IID (alpha=0.5) on full datasets
echo - Server Port: 8772
echo - Output: %WORK_DIR%\fl_mtl_3clients
echo ========================================
echo.
echo Multi-Task Learning with Non-IID creates optimal task specialization!
echo 3 clients per task with unique data distribution patterns.
echo.
pause
