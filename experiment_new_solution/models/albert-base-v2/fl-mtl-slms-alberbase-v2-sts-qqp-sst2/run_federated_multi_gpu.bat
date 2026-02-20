@echo off
REM Multi-GPU Federated Learning Startup Script
REM This script ensures both GPUs are visible and starts the federated learning system

echo ========================================
echo Multi-GPU Federated Learning Startup
echo ========================================
echo.

REM Set CUDA to use both GPUs
set CUDA_VISIBLE_DEVICES=0,1
echo CUDA_VISIBLE_DEVICES set to: %CUDA_VISIBLE_DEVICES%

REM Check GPU availability
echo Checking GPU availability...
C:\Users\hunglq\miniconda3\envs\py312\python.exe -c "
import torch
print(f'Available GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)')
"

echo.
echo Current GPU Assignment Strategy:
echo - SST-2 Client: GPU 0 (currently busy, will share)
echo - QQP Client:   GPU 1 (currently free)
echo - STS-B Client: GPU 0 (currently busy, will share)
echo.

echo ========================================
echo Starting Federated Learning Server...
echo ========================================
start "Federated Server" cmd /k "cd /d %~dp0 && C:\Users\hunglq\miniconda3\envs\py312\python.exe federated_main.py --mode server --config federated_config.yaml"

timeout /t 5 /nobreak

echo ========================================
echo Starting SST-2 Client on GPU 0...
echo ========================================
start "SST-2 Client" cmd /k "cd /d %~dp0 && C:\Users\hunglq\miniconda3\envs\py312\python.exe federated_main.py --mode client --client_id sst2_client --tasks sst2 --config federated_config.yaml"

timeout /t 2 /nobreak

echo ========================================
echo Starting QQP Client on GPU 1...
echo ========================================
start "QQP Client" cmd /k "cd /d %~dp0 && C:\Users\hunglq\miniconda3\envs\py312\python.exe federated_main.py --mode client --client_id qqp_client --tasks qqp --config federated_config.yaml"

timeout /t 2 /nobreak

echo ========================================
echo Starting STS-B Client on GPU 0...
echo ========================================
start "STS-B Client" cmd /k "cd /d %~dp0 && C:\Users\hunglq\miniconda3\envs\py312\python.exe federated_main.py --mode client --client_id stsb_client --tasks stsb --config federated_config.yaml"

echo.
echo ========================================
echo All processes started!
echo ========================================
echo.
echo GPU Usage Distribution:
echo - GPU 0: Server + SST-2 Client + STS-B Client
echo - GPU 1: QQP Client (dedicated)
echo.
echo Check individual windows for progress and logs.
echo Results will be saved to: federated_results\
echo.

pause
