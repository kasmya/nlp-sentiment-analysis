import streamlit as st
import pandas as pd
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from scipy.special import softmax
import nltk
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# Page config
st.set_page_config(page_title="Sentiment Analysis App", layout="wide")

# Model names
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

@st.cache_resource
def load_models():
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('punkt', quiet=True)
    sia = SentimentIntensityAnalyzer()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    return sia, tokenizer, model

def vader_score(text):
    sia = load_models()[0]
    return sia.polarity_scores(text)

def roberta_score(text):
    _, tokenizer, model = load_models()
    encoded = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    output = model(**encoded)
    scores = softmax(output[0][0].detach().numpy())
    return {
        'roberta_neg': float(scores[0]),
        'roberta_neu': float(scores[1]),
        'roberta_pos': float(scores[2])
    }

st.title("🤖 Amazon Review Sentiment Analyzer")
st.markdown("Compare **VADER** (dictionary-based) vs **RoBERTa** (AI model) on reviews!")

# Sidebar
st.sidebar.header("Options")
single_mode = st.sidebar.radio("Mode", ["Single Review", "Upload CSV"])

if single_mode == "Single Review":
    text = st.text_area("Enter review text:", height=200, placeholder="I love this product! Best purchase ever...")
    if st.button("Analyze", type="primary"):
        if text:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("VADER Scores")
                vader = vader_score(text)
                st.json(vader)
                label = "POSITIVE" if vader['compound'] >= 0.05 else "NEGATIVE" if vader['compound'] <= -0.05 else "NEUTRAL"
                st.success(f"**{label}** (Compound: {vader['compound']:.3f})")
            
            with col2:
                st.subheader("RoBERTa Scores")
                roberta = roberta_score(text)
                st.json(roberta)
                label = "POSITIVE" if roberta['roberta_pos'] > 0.5 else "NEGATIVE" if roberta['roberta_neg'] > 0.5 else "NEUTRAL"
                st.success(f"**{label}** (Max: {max(roberta['roberta_pos'], roberta['roberta_neg']):.3f})")
            
            # Comparison chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name="VADER", x=['Neg', 'Neu', 'Pos'], y=[vader['neg'], vader['neu'], vader['pos']]))
            fig.add_trace(go.Bar(name="RoBERTa", x=['Neg', 'Neu', 'Pos'], y=[roberta['roberta_neg'], roberta['roberta_neu'], roberta['roberta_pos']]))
            fig.update_layout(barmode='group', title="VADER vs RoBERTa")
            st.plotly_chart(fig, use_container_width=True)

else:  # CSV Mode
    uploaded = st.file_uploader("Upload Reviews.csv", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        if 'Text' in df.columns:
            if st.button("Analyze Dataset"):
                progress = st.progress(0)
                results = []
                for i in range(len(df)):
                    text = df.iloc[i]['Text']
                    vader = vader_score(text)
                    roberta = roberta_score(text)
                    results.append({**vader, **roberta, 'text': text[:100]})
                    progress.progress((i+1)/len(df))
                
                result_df = pd.DataFrame(results)
                st.dataframe(result_df)
                
                # Download
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Results", csv, "sentiment_results.csv", "text/csv")
                
                # Visuals
                col1, col2 = st.columns(2)
                with col1:
                    fig1 = px.histogram(result_df, x='compound', nbins=20, title="VADER Compound Distribution")
                    st.plotly_chart(fig1, use_container_width=True)
                with col2:
                    fig2 = px.histogram(result_df, x='roberta_pos', title="RoBERTa Positive Scores")
                    st.plotly_chart(fig2, use_container_width=True)
        else:
            st.error("CSV must have 'Text' column!")

st.markdown("---")

if __name__ == "__main__":
    pass

