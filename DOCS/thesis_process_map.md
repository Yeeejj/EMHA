# INSIDE-OUT: Thesis Process Map

> Emotion Recognition through Handwriting Analysis using CNN-HMM Hybrid Approach
> Cyrel Jane A. Edaño — University of San Carlos, DCISM

## Process Flow (Mermaid)

```mermaid
flowchart TB
    subgraph PHASE1["Phase 1: Assessment Design"]
        A1["3-Part Instrument"]
        A1a["Part 1: 24-item Self-Report\n(Likert Scale 1-5)"]
        A1b["Part 2: 4 Drawing Exercises\n(Circles, Dots, Person, House)"]
        A1c["Part 3: Writing Exercises\n(5 words × 3 styles + 5 cursive)"]
        A1 --> A1a & A1b & A1c
    end

    subgraph PHASE2["Phase 2: Data Collection"]
        B1["Participant Registration\n300+ participants\nAge 18-25, right-handed"]
        B2["Barcode ID Assignment\nP001 → P300+"]
        B3["Handwriting & Drawing\nScanning → PNG"]
        B1 --> B2 --> B3
    end

    subgraph PHASE3["Phase 3: Scoring & Labeling"]
        C1["Questionnaire Scoring"]
        C2["DASS-21\nDepression subscale (0-42)"]
        C3["Happiness Scale\n(Oxford/SDHS)"]
        C4{"Labeling\nThresholds"}
        C5["HAPPY\nhappiness ≥ 40\nAND dass < 14"]
        C6["SAD\nhappiness < 30\nOR dass ≥ 14"]
        C7["NEUTRAL\n(excluded)"]
        C1 --> C2 & C3
        C2 & C3 --> C4
        C4 -->|High happiness\nLow depression| C5
        C4 -->|Low happiness\nOR high depression| C6
        C4 -->|Ambiguous| C7
    end

    subgraph PHASE4["Phase 4: Image Preprocessing"]
        D1["Grayscale Conversion"]
        D2["Binarization\n(Otsu's Method)"]
        D3["Noise Removal\n(Morphological Ops)"]
        D4["Skew Correction\n(Rotation)"]
        D5["Normalize\nResize → 224×224"]
        D1 --> D2 --> D3 --> D4 --> D5
    end

    subgraph PHASE5["Phase 5: CNN Feature Extraction"]
        E1["Conv2d(1→32) + BN + ReLU + MaxPool"]
        E2["Conv2d(32→64) + BN + ReLU + MaxPool"]
        E3["Conv2d(64→128) + BN + ReLU + MaxPool"]
        E4["Conv2d(128→256) + BN + ReLU"]
        E5["Spatial Sequence Extraction\nHeight-average → (batch, 28, 256)\nLeft-to-right progression"]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    subgraph PHASE6["Phase 6: HMM Classification"]
        F1["GaussianHMM per class\n4 states, diagonal covariance"]
        F2["HMM_HAPPY\ntrained on happy sequences"]
        F3["HMM_SAD\ntrained on sad sequences"]
        F4["Compare log-likelihoods\nSoftmax → confidence"]
        F1 --> F2 & F3
        F2 & F3 --> F4
    end

    subgraph PHASE7["Phase 7: Validation"]
        G1["Stratified 5-Fold\nCross-Validation"]
        G2["Per fold:\n1. Train CNN\n2. Extract sequences\n3. Train HMM\n4. Evaluate"]
        G1 --> G2
    end

    subgraph PHASE8["Phase 8: Evaluation"]
        H1["Accuracy\n(target: 80-85%)"]
        H2["Precision\n(macro-averaged)"]
        H3["Recall\n(macro-averaged)"]
        H4["F1-Score\n(macro-averaged)"]
        H5["Confusion Matrix"]
    end

    subgraph PHASE9["Phase 9: Final Model"]
        I1["cnn_model.pth"]
        I2["hmm_models.pkl"]
        I3["cv_results.json"]
        I4["Prediction Pipeline:\nImage → Preprocess → CNN → HMM → HAPPY/SAD"]
    end

    PHASE1 ==> PHASE2
    PHASE2 ==> PHASE3
    PHASE3 ==> PHASE4
    PHASE4 ==> PHASE5
    PHASE5 ==> PHASE6
    PHASE6 ==> PHASE7
    PHASE7 ==> PHASE8
    PHASE8 ==> PHASE9

    style PHASE1 fill:#2d2b55,stroke:#667eea,color:#fff
    style PHASE2 fill:#2d2b55,stroke:#f093fb,color:#fff
    style PHASE3 fill:#2d2b55,stroke:#4facfe,color:#fff
    style PHASE4 fill:#2d2b55,stroke:#43e97b,color:#fff
    style PHASE5 fill:#2d2b55,stroke:#fee140,color:#fff
    style PHASE6 fill:#2d2b55,stroke:#ff6b6b,color:#fff
    style PHASE7 fill:#2d2b55,stroke:#ffd93d,color:#fff
    style PHASE8 fill:#2d2b55,stroke:#6bcb77,color:#fff
    style PHASE9 fill:#2d2b55,stroke:#4d96ff,color:#fff
```

## Data Flow Summary

```
Assessment → Scan → Score → Label → Preprocess → CNN Features → HMM Classify → 5-Fold CV → Evaluate → HAPPY/SAD
```

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchvision |
| Sequence Modeling | hmmlearn (GaussianHMM) |
| Image Processing | OpenCV, Pillow |
| ML & Evaluation | scikit-learn |
| Data | pandas, numpy |
| Visualization | matplotlib, seaborn |
| Platform | Google Colab (GPU), Google Drive |
| Psychometrics | DASS-21, Oxford/SDHS |

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Image size | 224×224 grayscale |
| CNN features | 256-dimensional |
| HMM states | 4 per class |
| HMM covariance | Diagonal |
| Sequence length | 28 timesteps (spatial columns) |
| Batch size | 32 |
| Learning rate | 0.001 |
| Early stopping | patience = 10 |
| Cross-validation | 5-fold stratified |
| Target accuracy | 80-85% |
