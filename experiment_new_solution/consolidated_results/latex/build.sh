#!/bin/bash

# Define document name
DOC_NAME="RT-FedMTL-March26"

echo "🚀 Starting LaTeX Compilation for ${DOC_NAME}..."

# Clean previous builds
echo "🧹 Cleaning up intermediate files..."
rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz *.toc *.lot *.lof

# First pass: Generate .aux files
echo "🚀 Pass 1: Generating auxiliary files (pdflatex)..."
pdflatex -interaction=nonstopmode "${DOC_NAME}.tex" > /dev/null

# Generate bibliography: Processes citations in .aux files
if [ -f "${DOC_NAME}.aux" ]; then
    echo "📚 Pass 2: Generating bibliography (bibtex)..."
    bibtex "${DOC_NAME}" > /dev/null
fi

# Second pass: Associate citations with text
echo "🚀 Pass 3: Resolving citations (pdflatex)..."
pdflatex -interaction=nonstopmode "${DOC_NAME}.tex" > /dev/null

# Third pass: Finalize cross-references and page numbers
echo "🚀 Pass 4: Finalizing document (pdflatex)..."
pdflatex -interaction=nonstopmode "${DOC_NAME}.tex" > /dev/null

# Check if PDF exists
if [ -f "${DOC_NAME}.pdf" ]; then
    echo "========================================"
    echo "✅ Success! PDF created: ${DOC_NAME}.pdf"
    echo "📄 File size: $(ls -lh ${DOC_NAME}.pdf | awk '{print $5}')"
    echo "📅 Finished at: $(date)"
    echo "========================================"
else
    echo "❌ Error: PDF creation failed. Check ${DOC_NAME}.log for details."
    exit 1
fi
