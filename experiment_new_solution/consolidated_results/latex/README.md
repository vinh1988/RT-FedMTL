# LaTeX Document Structure

This directory contains the complete LaTeX document for the federated learning experiments paper.

## Files

### `main.tex`
- Main LaTeX document with abstract, keywords, conclusion, and bibliography
- Includes the experiments section as separate input file
- Ready for compilation with XeLaTeX or pdfLaTeX

### `experiments_section.tex`
- Complete Section 5: EXPERIMENTS following the specified requirements
- Includes all subsections: 5.1 Experimental Setup, 5.2 Results, 5.3 Discussion
- Integrated figures and tables from the plots directory
- Comprehensive analysis based on the experimental results

## Document Structure

```
main.tex
├── Abstract
├── Keywords
├── Introduction (placeholder)
├── experiments_section.tex
│   ├── 5.1 Experimental Setup
│   │   ├── Datasets and Tasks
│   │   ├── Baselines
│   │   ├── Evaluation Metrics
│   │   └── Implementation Details
│   ├── 5.2 Results
│   │   ├── Overall Multi-Task Performance
│   │   ├── Impact of Federated Learning
│   │   ├── Efficiency Analysis
│   │   └── Ablation Studies
│   └── 5.3 Discussion
│       ├── Trade-offs between Privacy, Performance, and Efficiency
│       ├── Insights into Task Transfer and Negative Transfer
│       └── Practical Deployment Considerations
├── Conclusion
└── Bibliography
```

## Key Features

### Integrated Visualizations
- All figures reference the balanced-enhanced plots from `../plots/`
- High-quality PNG images with 1.5x font sizes for optimal readability
- Proper figure captions and cross-references

### Comprehensive Tables
- Performance metrics extracted from markdown files
- Efficiency analysis with training time and resource usage
- Ablation study results with detailed comparisons

### Analysis Highlights
- **Performance**: Single-task outperforms multi-task in 85% of cases
- **Federated Learning**: 2.4-3.9x training overhead but 77-93% resource savings
- **Model Selection**: distil-bert provides optimal balance
- **Task Transfer**: Negative transfer effects most pronounced in STSB task

## Compilation

To compile the document:

```bash
xelatex main.tex
# or
pdflatex main.tex
```

## Dependencies

- IEEE conference document class
- Standard LaTeX packages: graphicx, booktabs, amsmath, etc.
- All plot files should be in the `../plots/` directory

## Content Summary

The document provides a comprehensive analysis of:

1. **Experimental Setup**: Datasets (SST-2, QQP, STS-B), baselines, metrics
2. **Results**: Performance comparisons, federated learning impact, efficiency analysis
3. **Discussion**: Trade-offs, transfer learning insights, deployment considerations

All analysis is based on the experimental data from the plots and markdown files in the `../plots/` directory.
