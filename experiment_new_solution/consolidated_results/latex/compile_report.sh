#!/bin/bash

# Compilation script for NLU Federated Multi-Task Learning Paper
# Target: LaTeX directory

echo "📄 Starting LaTeX Compilation..."

# Check if pdflatex is installed
if ! command -v pdflatex &> /dev/null; then
    echo "❌ Error: pdflatex is not installed. Please install a LaTeX distribution (e.g., TeX Live)."
    exit 1
fi

# Cleanup old build files
echo "🧹 Cleaning up intermediate files..."
rm -f *.aux *.log *.out *.toc *.pdf

# Multi-pass compilation for references and tables
echo "🚀 Compiling main.tex (Pass 1)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

echo "🚀 Compiling main.tex (Pass 2)..."
pdflatex -interaction=nonstopmode main.tex > /dev/null

if [ -f "main.pdf" ]; then
    echo "========================================"
    echo "✅ Success! Paper compiled: main.pdf"
    echo "📅 Finished at: $(date)"
    echo "========================================"
else
    echo "❌ Error: PDF compilation failed. Check main.log for details."
    exit 1
fi
