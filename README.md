README: Sentiment Analysis with Pre-trained BERT Model
Project Description
This project demonstrates a sentiment analysis tool using the pre-trained nlptown/bert-base-multilingual-uncased-sentiment model from Hugging Face's Transformers library. The model classifies text input into three sentiment categories: positive, neutral, and negative.

Features
Text Sentiment Classification: Analyzes input text and determines whether it conveys a positive, neutral, or negative sentiment.
Confidence Scoring: Displays the confidence percentage for the predicted sentiment.
Interactive User Input: Accepts text input interactively and processes it until the user exits.
Installation
Prerequisites
Ensure you have Python 3.7 or higher installed on your system.

Dependencies
Install the required libraries using:

bash
Copy code
pip install transformers torch
How to Use
1. Run the Script
Execute the script by running:

bash
Copy code
python sentiment_analysis.py
2. Input Text
Provide a text input when prompted:

plaintext
Copy code
Enter a statement (or type 'exit' to quit): This is an amazing tool!
Predicted Sentiment: positive (Confidence: 98.75%)
3. Exit the Program
Type exit to terminate the program:

plaintext
Copy code
Enter a statement (or type 'exit' to quit): exit
Exiting the program.
Code Explanation
1. Model and Tokenizer Loading
The script uses the nlptown/bert-base-multilingual-uncased-sentiment model, which is pre-trained for multilingual sentiment classification. The tokenizer converts the text into a format suitable for the BERT model.

python
Copy code
from transformers import BertTokenizer, BertForSequenceClassification

model_name = 'nlptown/bert-base-multilingual-uncased-sentiment'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
2. Sentiment Prediction Function
The predict_sentiment function takes a text input, tokenizes it, processes it through the model, and determines the sentiment label based on probabilities.

python
Copy code
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=-1)
    sentiment = torch.argmax(probs).item()
    if sentiment == 0 or sentiment == 1:
        sentiment_label = "negative"
    elif sentiment == 2:
        sentiment_label = "neutral"
    else:
        sentiment_label = "positive"
    return sentiment_label, probs[0][sentiment].item()
3. Interactive Input Loop
The program allows users to input text iteratively. Sentiment analysis results are displayed, including a confidence score.

python
Copy code
while True:
    user_input = input("Enter a statement (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting the program.")
        break
    sentiment, confidence = predict_sentiment(user_input)
    print(f"Predicted Sentiment: {sentiment} (Confidence: {confidence*100:.2f}%)")
Output Example
Input:
plaintext
Copy code
Enter a statement (or type 'exit' to quit): I love programming!
Output:
plaintext
Copy code
Predicted Sentiment: positive (Confidence: 97.84%)
