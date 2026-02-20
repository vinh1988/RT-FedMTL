@echo off
REM Enhanced Federated Learning Multi-Task Learning (MTL) with Non-IID Distribution
REM Medium BERT with 3 clients per task (9 total clients)

echo ========================================
echo FL-MTL Non-IID 3-Client-Each Setup
echo ========================================
echo.

REM Set environment variables
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS\experiment_new_solution\models\medium-bert\fl-mtl-slms-medium-bert-non-iid-sst2-qqp-sst2-3client-each
set WORK_DIR=%PROJECT_ROOT%

REM Check if conda environment exists
echo Checking conda environment...
%CONDA_ROOT%\condabin\conda.bat env list | findstr %CONDA_ENV%
if %errorlevel% neq 0 (
    echo ERROR: Conda environment '%CONDA_ENV%' not found!
    echo Please create it first: conda create -n %CONDA_ENV% python=3.12
    pause
    exit /b 1
)

echo Conda environment '%CONDA_ENV%' found
echo Project directory: %WORK_DIR%
echo.

REM Start server first
echo Starting Federated Server (Medium BERT MTL Non-IID)...
start "FL-MTL Server" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode server --port 8773"

echo Server started on port 8773
echo Waiting 5 seconds for server to initialize...
timeout /t 5 >nul

REM ===== SST2 Clients =====
echo Starting SST2 Client 1 (Medium BERT Non-IID)...
start "FL-MTL SST2 Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_1 --tasks sst2 --port 8773"

timeout /t 3 >nul

echo Starting SST2 Client 2 (Medium BERT Non-IID)...
start "FL-MTL SST2 Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_2 --tasks sst2 --port 8773"

timeout /t 3 >nul

echo Starting SST2 Client 3 (Medium BERT Non-IID)...
start "FL-MTL SST2 Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id sst2_client_3 --tasks sst2 --port 8773"

timeout /t 3 >nul

REM ===== QQP Clients =====
echo Starting QQP Client 1 (Medium BERT Non-IID)...
start "FL-MTL QQP Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_1 --tasks qqp --port 8773"

timeout /t 3 >nul

echo Starting QQP Client 2 (Medium BERT Non-IID)...
start "FL-MTL QQP Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_2 --tasks qqp --port 8773"

timeout /t 3 >nul

echo Starting QQP Client 3 (Medium BERT Non-IID)...
start "FL-MTL QQP Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id qqp_client_3 --tasks qqp --port 8773"

timeout /t 3 >nul

REM ===== STSB Clients =====
echo Starting STSB Client 1 (Medium BERT Non-IID)...
start "FL-MTL STSB Client 1" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_1 --tasks stsb --port 8773"

timeout /t 3 >nul

echo Starting STSB Client 2 (Medium BERT Non-IID)...
start "FL-MTL STSB Client 2" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_2 --tasks stsb --port 8773"

timeout /t 3 >nul

echo Starting STSB Client 3 (Medium BERT Non-IID)...
start "FL-MTL STSB Client 3" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python federated_main.py --mode client --client_id stsb_client_3 --tasks stsb --port 8773"

echo.
echo ========================================
echo All clients started successfully!
echo ========================================
echo.
echo Server: Port 8773
echo Clients: 9 total (3 SST2 + 3 QQP + 3 STSB)
echo Model: prajjwal1/bert-medium (BERT-Medium, 8 layers, 512 hidden, ~41M params)
echo Distribution: Non-IID (alpha=0.5)
echo.
echo Check individual client windows for progress
echo Results will be saved to: fl_mtl_non-iid-3clients
echo.
pause
