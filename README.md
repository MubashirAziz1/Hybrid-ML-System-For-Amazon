# Amazon Product Return Risk Predictor

This project implements a hybrid machine learning system to predict the likelihood of product returns based on Amazon product reviews. The system combines text analysis with structured features to provide accurate return risk predictions.

## Data

The project uses the Amazon Product Reviews dataset. You can obtain the data in two ways:

1. **Direct Download**:
   - Download the Amazon Product Reviews dataset from [Kaggle](https://www.kaggle.com/datasets/PromptCloudHQ/amazon-reviews-unlocked-mobile-phones)
   - Place the downloaded `Amazon_Unlocked_Mobile.csv` file in the `data/` directory

2. **Using Kaggle API**:
   ```bash
   # Install Kaggle API
   pip install kaggle

   # Configure Kaggle API (follow instructions at https://github.com/Kaggle/kaggle-api)
   # Download the dataset
   kaggle datasets download -d PromptCloudHQ/amazon-reviews-unlocked-mobile-phones
   unzip amazon-reviews-unlocked-mobile-phones.zip
   mv Amazon_Unlocked_Mobile.csv data/
   ```

3. **Alternative Dataset**:
   If you want to use a different Amazon reviews dataset, ensure it has the following columns:
   - `Reviews`: Text content of the review
   - `Rating`: Numerical rating (1-5)
   - `Price`: Product price

   You can modify the data loading code in `src/data_prep.py` to match your dataset's structure.

## Features

- Text-based analysis using DistilBERT for sentiment and return intent detection
- Tabular feature analysis using XGBoost for structured data
- Hybrid model combining both approaches using a neural network fusion layer
- Interactive Streamlit dashboard for visualization and predictions
- SHAP-based feature explanations for model interpretability
- Keyword impact analysis for understanding return triggers

## Project Structure

```
├── data/                  # Data directory
│   └── Amazon_Unlocked_Mobile.csv  # Place your data file here
├── src/                   # Source code
│   ├── data_prep.py      # Data preprocessing and feature extraction
│   ├── text_model.py     # DistilBERT-based text classification
│   ├── tabular_model.py  # XGBoost model for structured features
│   ├── hybrid_model.py   # Neural network fusion of both models
│   └── dashboard.py      # Streamlit dashboard for visualization
├── notebooks/            # Jupyter notebooks for analysis
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/amazon-return-predictor.git
cd amazon-return-predictor
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset (see Data section above)

5. Run the Streamlit dashboard:
```bash
streamlit run src/dashboard.py
```

## Running on Kaggle/Google Colab

To run this project on Kaggle or Google Colab:

1. Create a new notebook
2. Clone the repository:
```python
!git clone https://github.com/yourusername/amazon-return-predictor.git
%cd amazon-return-predictor
```

3. Install dependencies:
```python
!pip install -r requirements.txt
```

4. Download the dataset:
```python
# Using Kaggle API (if running on Kaggle)
!kaggle datasets download -d PromptCloudHQ/amazon-reviews-unlocked-mobile-phones
!unzip amazon-reviews-unlocked-mobile-phones.zip
!mkdir -p data
!mv Amazon_Unlocked_Mobile.csv data/
```

5. Run the training and evaluation:
```python
from src.data_prep import prepare_data
from src.text_model import TextModel
from src.tabular_model import TabularModel
from src.hybrid_model import HybridPredictor

# Load and prepare data
X_train, X_test, y_train, y_test, df = prepare_data(
    'data/Amazon_Unlocked_Mobile.csv',
    sample_size=1000  # Adjust sample size as needed
)

# Train and evaluate models
text_model = TextModel()
tabular_model = TabularModel()
hybrid_model = HybridPredictor()

# Train models
text_model.train(text_model.prepare_data(X_train, y_train, X_test, y_test)[0])
tabular_model.train(X_train, y_train)
hybrid_model.train(X_train, y_train, X_test, y_test)

# Evaluate models
text_metrics = text_model.evaluate(text_model.prepare_data(X_train, y_train, X_test, y_test)[1])
tabular_metrics = tabular_model.evaluate(X_test, y_test)
hybrid_metrics = hybrid_model.evaluate(X_test, y_test)

print("Text Model Performance:", text_metrics)
print("Tabular Model Performance:", tabular_metrics)
print("Hybrid Model Performance:", hybrid_metrics)
```

## Usage

1. Data Preparation:
   - Place your Amazon review dataset in the `data/` directory
   - The dataset should be in CSV format with columns: Reviews, Rating, Price

2. Model Training:
   - The models will be trained automatically when running the dashboard
   - Training progress and metrics will be displayed in the console

3. Making Predictions:
   - Use the Streamlit dashboard to input new reviews
   - Get return risk predictions and explanations
   - View model performance metrics and feature importance

## Model Performance

The hybrid model combines the strengths of both text and tabular models:
- Text Model: Captures sentiment and return intent from review text
- Tabular Model: Handles structured features like price and rating
- Hybrid Model: Provides improved accuracy through model fusion

## Future Improvements

- Add more product categories
- Implement real-time prediction API
- Add more advanced feature engineering
- Enhance visualization capabilities
- Add model versioning and experiment tracking
- Implement automated testing
- Add CI/CD pipeline

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Amazon for providing the review dataset
- Hugging Face for the DistilBERT model
- Streamlit for the dashboard framework 