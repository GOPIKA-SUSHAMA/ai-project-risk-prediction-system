# AI Project Risk Prediction System (Multi-Output ML)

This repository implements a **multi-output AI pipeline** for predicting and managing project risks, inspired by the paper:

**Geamanu et al. (2025)** — *An Integrated Artificial Intelligence Tool for Predicting and Managing Project Risks* (Machine Learning and Knowledge Extraction, MDPI).

 ## What this project does
Given structured project attributes (budget variance, technical complexity, regulatory impact, stakeholder influence, etc.), the system predicts **four risk outputs**:

- **Risk Type** (Classification): Technical / Financial / Compliance / Operational  
- **Risk Response Strategy** (Classification): Accept / Mitigate / Transfer / Avoid  
- **Risk Impact** (Regression-like integer score)
- **Risk Probability** (Regression-like percentage)

## Dataset
The dataset is **synthetic and rule-based** (5,000 rows, 27+ features).  
Targets are deterministically derived using explicit rules—useful for validating an end-to-end multi-output pipeline when real project-risk data is confidential.

## Models
- Decision Tree (baseline, interpretable)
- Random Forest (ensemble)

Classification is trained with `MultiOutputClassifier` and regression with `MultiOutputRegressor`.

## Repository structure
```
ai-project-risk-prediction-system/
├── notebooks/
│   └── 01_risk_prediction_multioutput.ipynb
├── src/
│   ├── generate_dataset.py
│   ├── train_models.py
│   └── evaluate.py
├── data/            # generated CSV (optional)
├── models/          # trained models (joblib)
├── docs/
│   └── Final_Report_Gopika_Sushamakumari_2560644.pdf
├── requirements.txt
└── .gitignore
```

## How to run (local)
**1) Create environment & install dependencies**
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

**2) Train models**
```bash
cd src
python train_models.py
```

**3) Evaluate**
```bash
python evaluate.py
```

## Notes on results
Because the targets are rule-based, tree models can achieve very high performance (sometimes near-perfect).  
In a real deployment, you would replace the synthetic generator with real (or semi-synthetic) project datasets and add explainability (e.g., SHAP) for stakeholder trust.

## Author
**Gopika Sushamakumari**

**Researcher AI**
