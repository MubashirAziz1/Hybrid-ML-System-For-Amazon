import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
from text_model import TextModel
from tabular_model import TabularModel
from hybrid_model import HybridModel
import json

def load_models():
    """Load all trained models."""
    try:
        # Load text model
        text_model = TextModel()
        text_model.load('models/text_model.pt')
        
        # Load tabular model
        tabular_model = TabularModel()
        tabular_model.load('models/tabular_model.joblib')
        
        # Load hybrid model
        hybrid_model = HybridModel()
        hybrid_model.load('models/hybrid_model.joblib')
        
        return text_model, tabular_model, hybrid_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

def create_model_comparison_plot(metrics):
    """Create a comparison plot of model performances."""
    models = ['Text Model', 'Tabular Model', 'Hybrid Model']
    f1_scores = [
        metrics['text_metrics']['f1_score'],
        metrics['tabular_metrics']['f1_score'],
        metrics['hybrid_metrics']['f1_score']
    ]
    accuracies = [
        metrics['text_metrics']['accuracy'],
        metrics['tabular_metrics']['accuracy'],
        metrics['hybrid_metrics']['accuracy']
    ]
    
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

def create_feature_importance_plot(model, X_test):
    """Create a feature importance plot."""
    importance = model.get_feature_importance(X_test)
    
    fig = px.bar(
        importance,
        x='feature',
        y='importance',
        title='Feature Importance (SHAP Values)'
    )
    
    return fig

def predict_return_risk(review_text, price, rating, model_type):
    """Predict return risk using the selected model."""
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'cleaned_review': [review_text],
            'normalized_price': [price],
            'Rating': [rating],
            'review_length': [len(review_text)]
        })
        
        # Get prediction based on model type
        if model_type == "Text Model":
            prob = text_model.predict(input_data)[0]
        elif model_type == "Tabular Model":
            prob = tabular_model.predict(input_data)[0]
        else:  # Hybrid Model
            prob = hybrid_model.predict(input_data, tabular_model, text_model)[0]
        
        # Format prediction
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
        return f"Return Risk: {risk_level} ({prob:.2%})"
    except Exception as e:
        return f"Error making prediction: {str(e)}"

def create_data_overview(df):
    """Create data overview plots."""
    # Rating distribution
    rating_fig = px.histogram(df, x='Rating', title='Rating Distribution')
    
    # Price distribution
    price_fig = px.box(df, y='Price', title='Price Distribution')
    
    return rating_fig, price_fig

def create_dashboard():
    """Create the Gradio interface."""
    # Load models
    global text_model, tabular_model, hybrid_model
    text_model, tabular_model, hybrid_model = load_models()
    
    if text_model is None or tabular_model is None or hybrid_model is None:
        return gr.Interface(
            fn=lambda x: "Error: Models could not be loaded. Please ensure models are trained first.",
            inputs=gr.Textbox(label="Error"),
            outputs=gr.Textbox(label="Status")
        )
    
    # Create interface
    interface = gr.Interface(
        fn=predict_return_risk,
        inputs=[
            gr.Textbox(label="Review Text"),
            gr.Number(label="Price"),
            gr.Number(label="Rating (1-5)"),
            gr.Radio(["Text Model", "Tabular Model", "Hybrid Model"], label="Model Type")
        ],
        outputs=gr.Textbox(label="Prediction"),
        title="Amazon Review Return Risk Predictor",
        description="Enter a review and product details to predict return risk."
    )
    
    return interface

if __name__ == "__main__":
    # Create and launch dashboard
    dashboard = create_dashboard()
    dashboard.launch() 