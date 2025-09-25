#!/bin/bash
# Ubuntu activation helper for FedBERT-LoRA

echo "🐧 Activating FedBERT-LoRA environment on Ubuntu..."
source venv/bin/activate

echo "✅ Environment activated!"
echo "System: $(lsb_release -d | cut -f2)"
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
else
    echo "Device: CPU"
fi

echo ""
echo "Quick start commands:"
echo "  python examples/run_simple_experiment.py"
echo "  python examples/run_glue_experiment.py --task sst2"
echo "  ./run_experiment.sh 5 10 sst2"
