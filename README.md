# Amazon Reviews Sentiment Analysis Notebook

## Overview
This Jupyter notebook (`sentiment_analysis.ipynb`) performs **sentiment analysis** on a sample of Amazon product reviews from the dataset `Reviews.csv`. It compares two popular approaches:

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: A rule-based lexicon and grammar tool for sentiment scoring.
2. **RoBERTa (Robustly optimized BERT approach)**: A transformer-based pre-trained model fine-tuned for sentiment analysis on tweets (twitter-roberta-base-sentiment).

The goal is to analyze review texts, compute sentiment scores (positive, neutral, negative, compound), visualize distributions by star ratings, and compare model performances.

## Dataset
- **Source**: Amazon Reviews CSV (`/Users/kasmyabhatia/Downloads/Reviews.csv`)
- **Used**: First 500 rows for efficiency.
- **Columns used**:
  - `Id`: Unique review ID
  - `Score`: Star rating (1-5)
  - `Text`: Review text

## Key Steps & Code Sections

### 1. Imports & Setup
```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm.notebook import tqdm
from scipy.special import softmax
```
- Loads data analysis, NLP, and ML libraries.
- Applies 'ggplot' style to plots.

### 2. Data Loading & EDA
- Loads CSV and limits to 500 rows.
- Bar plot: Review count by stars (shows class imbalance).

### 3. NLTK Basics (Exploratory)
- Tokenization, POS tagging, Named Entity Recognition (NER) on an example review.
- Demonstrates basic NLP preprocessing (not used in core analysis).

### 4. VADER Sentiment Analysis
- Initializes `SentimentIntensityAnalyzer`.
- Computes scores: `pos`, `neu`, `neg`, `compound` for each review.
- Merges results with original DF.
- **Visualizations**:
  - Bar: Compound score by stars.
  - Subplots: Pos/Neu/Neg distributions by stars.

### 5. RoBERTa Pre-trained Model
- Loads tokenizer and model: `cardiffnlp/twitter-roberta-base-sentiment`.
- Defines `polarity_scores_roberta()`: Tokenizes text, runs inference, applies softmax.
- Computes RoBERTa scores (`roberta_neg`, `roberta_neu`, `roberta_pos`) for all reviews.
- Handles runtime errors gracefully.

### 6. Comparison & Insights
- Merges VADER + RoBERTa results.
- **Pairplot**: Scatter matrix of all sentiment scores, colored by `Score`.
- Examples: Worst positive misclassifications (reviews with low stars but high model pos score).

### 7. Transformers Pipeline (Bonus)
- Quick demo of Hugging Face `pipeline(\"sentiment-analysis\")`.
- Install note: `!pip install transformers`.

## Dependencies
Run in Jupyter/Colab:
```
pip install pandas numpy matplotlib seaborn nltk transformers torch tqdm scipy
```
- NLTK data: `nltk.download('maxent_ne_chunker_tab')`, `nltk.download('words')`
- Note: RoBERTa requires PyTorch (auto-installed via transformers).

## Visualizations Generated
1. Review counts by stars.
2. VADER compound by stars.
3. VADER pos/neu/neg subplots.
4. Pairplot of VADER + RoBERTa scores by true stars.

## Key Findings (from plots/code)
- Higher star ratings correlate with more positive sentiment (as expected).
- RoBERTa may handle nuances better than VADER (inspect pairplot).
- Misclassifications: Some 1-star reviews scored positive by models (sarcasm?).

## How to Run
1. Place `Reviews.csv` in `/Users/kasmyabhatia/Downloads/` or update path.
2. Open `sentiment_analysis.ipynb` in Jupyter/VSCode.
3. Run all cells sequentially.
4. View plots inline.

## Potential Improvements
- Full dataset processing (remove `.head(500)`).
- Cross-validation with more metrics (accuracy, F1).
- Train custom model.
- Handle long texts (truncate for RoBERTa).

## References
- [Amazon Reviews Dataset](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)
- [VADER](https://github.com/cjhutto/vaderSentiment)
- [CardiffNLP Twitter-RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
