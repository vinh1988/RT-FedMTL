@echo off
echo ========================================
echo Centralized Training - Single Task
echo DistilBERT with STSB Dataset
echo ========================================

REM ===== Conda =====
set CONDA_ROOT=C:\Users\hunglq\miniconda3
set CONDA_ENV=py312

REM ===== Project paths =====
set PROJECT_ROOT=C:\Users\hunglq\docs\FedAvgLS
set WORK_DIR=%PROJECT_ROOT%\experiment_new_solution\models\distil-bert\centralized-single-task-stsb

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

REM ===== Start Centralized STSB Training =====
echo Starting Centralized STSB Training...
start "Centralized STSB Training" cmd /k ^
"%CONDA_ROOT%\condabin\conda.bat activate %CONDA_ENV% ^&^& cd /d %WORK_DIR% ^&^& set CUDA_VISIBLE_DEVICES=0 ^&^& python centralized_main.py"

timeout /t 3 >nul

echo.
echo ========================================
echo Centralized STSB training started!
echo ========================================
echo.
echo Model: distilbert-base-uncased
echo Dataset: STSB (4,249 train + 1,500 val samples)
echo Training Type: Centralized Single Task
echo.
echo Check training window for progress
echo Results will be saved to: centralized_stsb_results
echo.
pause
