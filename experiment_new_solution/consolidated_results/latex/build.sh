#!/bin/bash

# Build script for LaTeX document
echo "Building LaTeX document..."

# Clean previous builds
rm -f *.aux *.log *.out *.bbl *.blg *.synctex.gz

# First pass
echo "First compilation pass..."
xelatex -interaction=nonstopmode main.tex

# Second pass for cross-references
echo "Second compilation pass..."
xelatex -interaction=nonstopmode main.tex

# Check if PDF was created
if [ -f "main.pdf" ]; then
    echo "✅ PDF created successfully: main.pdf"
    echo "📄 File size: $(ls -lh main.pdf | awk '{print $5}')"
    
    # Open PDF if running on Linux with display
    if command -v xdg-open &> /dev/null && [ -n "$DISPLAY" ]; then
        xdg-open main.pdf
    fi
else
    echo "❌ PDF creation failed. Check for errors in main.log"
fi
