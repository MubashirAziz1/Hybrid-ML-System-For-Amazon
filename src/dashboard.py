import gradio as gr
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
import os

def load_data_and_models():
    """Load data and train models."""
    X_train, X_test, y_train, y_test, df = prepare_data(
        'data/Amazon_Unlocked_Mobile.csv',
        sample_size=1000
    )
    
    # Initialize models
    text_model = TextModel()
    tabular_model = TabularModel()
    hybrid_model = HybridPredictor()
    
    # Train models
    text_model.train(text_model.prepare_data(X_train, y_train, X_test, y_test)[0])
    tabular_model.train(X_train, y_train)
    hybrid_model.train(X_train, y_train, X_test, y_test)
    
    return X_train, X_test, y_train, y_test, df, text_model, tabular_model, hybrid_model

def create_model_comparison_plot(text_metrics, tabular_metrics, hybrid_metrics):
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

def predict_return_risk(review_text, rating, price):
    """Make predictions using the hybrid model."""
    # Prepare input data
    input_data = pd.DataFrame({
        'cleaned_review': [review_text],
        'Rating': [rating],
        'Price': [price]
    })
    
    # Get predictions
    hybrid_prob = hybrid_model.predict(input_data)[0][1]
    
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
    
    return f"Return Risk Probability: {hybrid_prob:.2%}", fig

def create_data_overview(df):
    """Create data overview plots."""
    # Rating distribution
    rating_fig = px.histogram(df, x='Rating', title='Rating Distribution')
    
    # Price distribution
    price_fig = px.box(df, y='Price', title='Price Distribution')
    
    return rating_fig, price_fig

# Load data and models
print("Loading data and training models...")
X_train, X_test, y_train, y_test, df, text_model, tabular_model, hybrid_model = load_data_and_models()

# Get model metrics
text_metrics = text_model.evaluate(text_model.prepare_data(X_train, y_train, X_test, y_test)[1])
tabular_metrics = tabular_model.evaluate(X_test, y_test)
hybrid_metrics = hybrid_model.evaluate(X_test, y_test)

# Create Gradio interface
with gr.Blocks(title="Amazon Return Risk Predictor") as demo:
    gr.Markdown("# Amazon Return Risk Predictor")
    gr.Markdown("This dashboard showcases the performance of our hybrid model for predicting product return risk based on Amazon reviews.")
    
    with gr.Tab("Model Performance"):
        gr.Markdown("## Model Performance Comparison")
        model_comparison = create_model_comparison_plot(text_metrics, tabular_metrics, hybrid_metrics)
        gr.Plot(model_comparison)
        
        gr.Markdown("## Feature Importance")
        feature_importance = create_feature_importance_plot(tabular_model, X_test)
        gr.Plot(feature_importance)
    
    with gr.Tab("Make Predictions"):
        gr.Markdown("## Predict Return Risk")
        with gr.Row():
            with gr.Column():
                review_input = gr.Textbox(
                    label="Product Review",
                    placeholder="Enter the product review here...",
                    lines=5
                )
                rating_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="Product Rating"
                )
                price_input = gr.Number(
                    label="Product Price ($)",
                    value=100.0
                )
                predict_btn = gr.Button("Predict Return Risk")
            
            with gr.Column():
                result_text = gr.Textbox(label="Prediction Result")
                result_plot = gr.Plot(label="Risk Score Gauge")
        
        predict_btn.click(
            fn=predict_return_risk,
            inputs=[review_input, rating_input, price_input],
            outputs=[result_text, result_plot]
        )
    
    with gr.Tab("Data Overview"):
        gr.Markdown("## Data Overview")
        rating_fig, price_fig = create_data_overview(df)
        gr.Plot(rating_fig)
        gr.Plot(price_fig)
        
        gr.Markdown("## Sample Reviews")
        sample_data = df[['Reviews', 'Rating', 'Price', 'return_risk']].head()
        gr.Dataframe(sample_data)

if __name__ == "__main__":
    demo.launch(share=True) 