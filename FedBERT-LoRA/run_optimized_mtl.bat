@echo off
REM 🚀 Optimized MTL Federated Learning System - Windows Startup Script
REM This batch file provides all commands needed to run the server and clients

echo 🚀 Optimized MTL Federated Learning System
echo ==========================================
echo.
echo 📋 Available Commands:
echo.

REM Colors for better output (Windows 10+)
echo 1. Environment Setup:
echo    cd /d "C:\path\to\FedBERT-LoRA"
echo    venv\Scripts\activate
echo.

echo 2. Server Startup:
echo    python optimized_mtl_federated.py --mode server --rounds 5 --total_clients 3
echo    REM Starts the MTL Federated Server on port 8771
echo    REM Runs for 5 rounds, expects 3 clients
echo.

echo 3. Client Startup Commands:
echo    REM Command Prompt 1 - Client 1:
echo    python optimized_mtl_federated.py --mode client --client_id client_1 --tasks sst2 qqp stsb
echo.
echo    REM Command Prompt 2 - Client 2:
echo    python optimized_mtl_federated.py --mode client --client_id client_2 --tasks sst2 qqp stsb
echo.
echo    REM Command Prompt 3 - Client 3:
echo    python optimized_mtl_federated.py --mode client --client_id client_3 --tasks sst2 qqp stsb
echo.

echo 4. Testing Commands:
echo    REM Test the optimized system:
echo    python test_optimized_mtl.py
echo.
echo    REM Run optimization demo:
echo    python optimization_demo.py
echo.

echo 5. Alternative Server Commands:
echo    REM Quick test (2 rounds, 2 clients):
echo    python optimized_mtl_federated.py --mode server --rounds 2 --total_clients 2
echo.
echo    REM Production run (10 rounds, 5 clients):
echo    python optimized_mtl_federated.py --mode server --rounds 10 --total_clients 5
echo.
echo    REM Custom configuration:
echo    python optimized_mtl_federated.py --mode server --rounds 3 --total_clients 3 --port 8772
echo.

echo 6. Alternative Client Commands:
echo    REM Single task clients:
echo    python optimized_mtl_federated.py --mode client --client_id client_sst2 --tasks sst2
echo    python optimized_mtl_federated.py --mode client --client_id client_qqp --tasks qqp
echo    python optimized_mtl_federated.py --mode client --client_id client_stsb --tasks stsb
echo.
echo    REM Two-task clients:
echo    python optimized_mtl_federated.py --mode client --client_id client_clf --tasks sst2 qqp
echo    python optimized_mtl_federated.py --mode client --client_id client_reg --tasks stsb
echo.

echo 7. Monitoring Commands:
echo    REM Check system logs:
echo    type optimized_mtl_federated.log ^| more
echo.
echo    REM Monitor server output:
echo    REM (Run in separate command prompt after starting server)
echo.

echo 8. Cleanup Commands:
echo    REM Stop all processes:
echo    taskkill /f /im python.exe
echo.
echo    REM Clean log files:
echo    del optimized_mtl_federated.log
echo    rmdir /s /q no_lora_results
echo.

echo 9. Complete Workflow Example:
echo    REM Command Prompt 1 - Server:
echo    cd /d "C:\path\to\FedBERT-LoRA"
echo    venv\Scripts\activate
echo    python optimized_mtl_federated.py --mode server --rounds 3 --total_clients 3
echo.
echo    REM Command Prompt 2 - Client 1:
echo    cd /d "C:\path\to\FedBERT-LoRA"
echo    venv\Scripts\activate
echo    python optimized_mtl_federated.py --mode client --client_id client_1 --tasks sst2 qqp stsb
echo.
echo    REM Command Prompt 3 - Client 2:
echo    cd /d "C:\path\to\FedBERT-LoRA"
echo    venv\Scripts\activate
echo    python optimized_mtl_federated.py --mode client --client_id client_2 --tasks sst2 qqp stsb
echo.
echo    REM Command Prompt 4 - Client 3:
echo    cd /d "C:\path\to\FedBERT-LoRA"
echo    venv\Scripts\activate
echo    python optimized_mtl_federated.py --mode client --client_id client_3 --tasks sst2 qqp stsb
echo.

echo 10. Expected Output:
echo    ✅ Server: 'Starting optimized server on port 8771'
echo    ✅ Server: 'Waiting for clients... (0/1)'
echo    ✅ Client: 'Client client_1 registered successfully'
echo    ✅ Server: 'Starting training with X clients'
echo    ✅ All: Performance metrics and completion status
echo.

echo 11. Troubleshooting:
echo    🔧 If clients fail to connect:
echo       - Ensure server is running first
echo       - Check firewall settings (netsh advfirewall firewall add rule)
echo       - Verify port 8771 is available (netstat -an ^| find "8771")
echo       - Check system resources (RAM, CPU)
echo.
echo    🔧 If training is slow:
echo       - Reduce --samples parameter
echo       - Use --rounds 2 for testing
echo       - Monitor system resources with Task Manager
echo.
echo    🔧 If getting import errors:
echo       - Run: venv\Scripts\activate
echo       - Check Python path (echo %PYTHONPATH%)
echo       - Verify all dependencies installed (pip list)
echo.

echo 12. Performance Tips:
echo    💡 Use SSD storage for better I/O performance
echo    💡 Monitor RAM usage (each client ~500MB-1GB)
echo    💡 Close unnecessary applications during training
echo    💡 Use multiple Command Prompts for different clients
echo.

echo 🎯 Happy Federated Learning! 🚀
echo.
echo For more information, see:
echo    - OPTIMIZATION_REPORT.md
echo    - MTL_FEDERATED_IMPLEMENTATION.md
echo    - optimized_config.ini

pause
