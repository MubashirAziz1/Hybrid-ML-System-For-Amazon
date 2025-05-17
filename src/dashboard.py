import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data_prep import prepare_data
from text_model import TextModel
from tabular_model import TabularModel
from hybrid_model import HybridPredictor
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Amazon Return Risk Predictor", layout="wide")

@st.cache_data
def load_data():
    """Load and prepare the data."""
    X_train, X_test, y_train, y_test, df = prepare_data(
        '../Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    return X_train, X_test, y_train, y_test, df

@st.cache_resource
def load_models():
    """Load and train the models."""
    X_train, X_test, y_train, y_test, _ = load_data()
    
    # Initialize models
    text_model = TextModel()
    tabular_model = TabularModel()
    hybrid_model = HybridPredictor()
    
    # Train models
    text_model.train(text_model.prepare_data(X_train, y_train, X_test, y_test)[0])
    tabular_model.train(X_train, y_train)
    hybrid_model.train(X_train, y_train, X_test, y_test)
    
    return text_model, tabular_model, hybrid_model

def plot_model_comparison(text_metrics, tabular_metrics, hybrid_metrics):
    """Create a comparison plot of model performances."""
    models = ['Text Model', 'Tabular Model', 'Hybrid Model']
    f1_scores = [text_metrics['f1_score'], tabular_metrics['f1_score'], hybrid_metrics['f1_score']]
    accuracies = [text_metrics['accuracy'], tabular_metrics['accuracy'], hybrid_metrics['accuracy']]
    
    fig = go.Figure(data=[
        go.Bar(name='F1 Score', x=models, y=f1_scores),
        go.Bar(name='Accuracy', x=models, y=accuracies)
    ])
    
    fig.update_layout(
        title='Model Performance Comparison',
        barmode='group',
        yaxis_title='Score',
        showlegend=True
    )
    
    return fig

def plot_feature_importance(model, X_test):
    """Create a feature importance plot."""
    importance = model.get_feature_importance(X_test)
    
    fig = px.bar(
        importance,
        x='feature',
        y='importance',
        title='Feature Importance (SHAP Values)'
    )
    
    return fig

def main():
    st.title("Amazon Return Risk Predictor")
    st.write("This dashboard showcases the performance of our hybrid model for predicting product return risk based on Amazon reviews.")
    
    # Load data and models
    with st.spinner("Loading data and training models..."):
        X_train, X_test, y_train, y_test, df = load_data()
        text_model, tabular_model, hybrid_model = load_models()
    
    # Model Performance Section
    st.header("Model Performance")
    
    # Get model metrics
    text_metrics = text_model.evaluate(text_model.prepare_data(X_train, y_train, X_test, y_test)[1])
    tabular_metrics = tabular_model.evaluate(X_test, y_test)
    hybrid_metrics = hybrid_model.evaluate(X_test, y_test)
    
    # Plot model comparison
    st.plotly_chart(plot_model_comparison(text_metrics, tabular_metrics, hybrid_metrics))
    
    # Feature Importance Section
    st.header("Feature Importance")
    st.plotly_chart(plot_feature_importance(tabular_model, X_test))
    
    # Prediction Section
    st.header("Make Predictions")
    
    # Input form
    with st.form("prediction_form"):
        review_text = st.text_area("Enter product review:", height=100)
        rating = st.slider("Product Rating:", 1, 5, 3)
        price = st.number_input("Product Price ($):", min_value=0.0, value=100.0)
        
        submitted = st.form_submit_button("Predict Return Risk")
        
        if submitted:
            # Prepare input data
            input_data = pd.DataFrame({
                'cleaned_review': [review_text],
                'Rating': [rating],
                'Price': [price]
            })
            
            # Get predictions
            hybrid_prob = hybrid_model.predict(input_data)[0][1]
            
            # Display prediction
            st.subheader("Prediction Results")
            st.write(f"Return Risk Probability: {hybrid_prob:.2%}")
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=hybrid_prob * 100,
                title={'text': "Return Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            st.plotly_chart(fig)
    
    # Data Overview Section
    st.header("Data Overview")
    
    # Display sample data
    st.subheader("Sample Reviews")
    st.dataframe(df[['Reviews', 'Rating', 'Price', 'return_risk']].head())
    
    # Distribution plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Rating', title='Rating Distribution')
        st.plotly_chart(fig)
    
    with col2:
        fig = px.box(df, y='Price', title='Price Distribution')
        st.plotly_chart(fig)

if __name__ == "__main__":
    main() 