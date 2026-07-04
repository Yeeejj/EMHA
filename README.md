# EMHA

## INSIDE-OUT: Emotion Recognition through Handwriting Analysis using Hybrid Approach

### Overview

INSIDE-OUT is a research project focused on developing a system that classifies emotional states (happiness vs. sadness) from handwriting samples using a hybrid CNN-HMM (Convolutional Neural Network - Hidden Markov Model) architecture.

### Thesis Information

- **Title:** INSIDE-OUT: Emotion Recognition through Handwriting Analysis using Hybrid Approach
- **Author:** Cyrel Jane A. Edaño
- **Institution:** University of San Carlos, Department of Computer, Information Sciences and Mathematics (DCISM)
- **Date:** May 2025

### Research Objectives

1. **Primary Goal:** Develop a hybrid CNN-HMM model to classify happiness and sadness from handwriting images
2. **Target Accuracy:** 80-85% classification accuracy
3. **Sample Size:** 300+ right-handed participants aged 18-25

### Methodology

The system employs a two-stage hybrid approach:

1. **CNN Stage:** Extracts spatial features from preprocessed handwriting images
   - Image preprocessing: binarization, noise removal, skew correction, normalization
   - Feature extraction from handwriting characteristics (slant, spacing, baseline, pressure)

2. **HMM Stage:** Performs temporal sequence classification
   - Models the sequential nature of handwriting strokes
   - Classifies emotional states based on extracted features

### Key Handwriting Features Analyzed

| Feature | Happy Indicators | Sad Indicators |
|---------|------------------|----------------|
| Baseline | Rising | Descending |
| Slant | Right-leaning | Left-leaning or vertical |
| Pressure | Heavy, consistent | Light, inconsistent |
| Size | Large letters | Small letters |
| Spacing | Wide, open | Narrow, cramped |

### Theoretical Foundations

- **Basic Emotions Theory (Ekman, 1992, 1999):** Six universal emotions framework
- **EMOTHAW Database:** Novel database for emotional state recognition from handwriting
- **Graphology Principles:** Handwriting reflects personality and emotional states

### Project Structure

```
EMHA/
├── CITATIONS/           # Research papers and references
│   ├── SCALES/          # Happiness/depression questionnaires
│   └── *.pdf            # Academic papers
├── DOCS/                # Thesis documents and proposals
├── README.md            # This file
└── README_THESIS_INSTRUCTIONS.md
```

### Evaluation Metrics

- F1-score (macro-averaged)
- Stratified 5-fold cross-validation
- Confusion matrix analysis
- Precision and recall per emotion class

### References

Key citations supporting this research:

1. Likforman-Sulem et al. (2017) - EMOTHAW Database
2. Ekman (1992, 1999) - Basic Emotions Theory
3. Goumiri et al. (2023) - CNN-HMM Hybrid Model
4. Kedar et al. (2015) - Automatic Emotion Recognition through Handwriting
5. Bhattacharya et al. (2022) - TEmoDec Framework

For a complete list of citations, see `CITATIONS/THESIS_CITATIONS_SUMMARY.md`

### Status

Research phase - Data collection and model development in progress.

---

*University of San Carlos - Department of Computer, Information Sciences and Mathematics*
