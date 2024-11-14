# Sentiment Analysis using BERT (nlptown/bert-base-multilingual-uncased-sentiment)

## Overview
This project implements sentiment analysis using the **BERT model** from Hugging Face. The model, `nlptown/bert-base-multilingual-uncased-sentiment`, is fine-tuned to classify text into 5 sentiment categories: **negative**, **neutral**, and **positive**.

The application allows users to input a statement, and it will predict the sentiment of the text along with the confidence score. This model can process multilingual text, making it a useful tool for global applications.

## Key Features
- Predicts sentiment labels: **negative**, **neutral**, and **positive**.
- Displays confidence score for the predicted sentiment.
- Uses the Hugging Face BERT model `nlptown/bert-base-multilingual-uncased-sentiment`.
- Supports continuous user input for multiple sentiment predictions until "exit" is typed.

## Requirements
- Python 3.x
- Install the required dependencies using `pip`:

```bash
pip install torch transformers

How to Use
Clone the repository (or download the files).

bash
Copy code
git clone https://github.com/your-username/sentiment-analysis-bert.git
Install dependencies: Ensure you have Python 3 and the necessary libraries installed. You can install them using the following command:

bash
Copy code
pip install torch transformers
Run the script: Execute the sentiment analysis script:

bash
Copy code
python sentiment_analysis.py
