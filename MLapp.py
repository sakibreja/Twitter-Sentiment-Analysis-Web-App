from flask import Flask, request, jsonify, render_template
import joblib
from joblib import load
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load your pre-trained logistic regression model
model = load(r'C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\TWITTER SENTIMENT ANALYSIS\logistic_regression.joblib')
vectorizer = joblib.load(r'C:\Users\A3MAX SOFTWARE TECH\A VS CODE\11. CAPSTONE PROJECT_DEPLOYMENT\TWITTER SENTIMENT ANALYSIS\vectorizer1.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text_data = request.form.get('text')
    text_data = [text_data.lower()]  # Apply preprocessing if needed

    # Transform the input text data using the vectorizer
    text_data_transformed = vectorizer.transform(text_data)

    # Predict the sentiment using the pre-trained model
    prediction = model.predict(text_data_transformed)

    # Convert the prediction to human-readable sentiment
    sentiment = "Positive" if prediction[0] == 1 else "Negative"

    return render_template('result.html', sentiment=sentiment)


if __name__ == '__main__':
    app.run(debug=True)
