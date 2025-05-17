# Amazon Product Return Risk Predictor

This project implements a hybrid machine learning system to predict the likelihood of product returns based on Amazon product reviews. The system combines text analysis with structured features to provide accurate return risk predictions.

## Features

- Text-based analysis using DistilBERT
- Tabular feature analysis using XGBoost
- Hybrid model combining both approaches
- Interactive Streamlit dashboard
- SHAP-based feature explanations
- Keyword impact analysis

## Project Structure

```
├── data/                  # Data directory
├── src/                   # Source code
│   ├── data_prep.py      # Data preprocessing
│   ├── text_model.py     # Text classification model
│   ├── tabular_model.py  # Tabular features model
│   ├── hybrid_model.py   # Combined model
│   └── dashboard.py      # Streamlit dashboard
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

## Model Performance

- Text-only model: [Performance metrics]
- Tabular-only model: [Performance metrics]
- Hybrid model: [Performance metrics]

## Future Improvements

- Add more product categories
- Implement real-time prediction API
- Add more advanced feature engineering
- Enhance visualization capabilities 