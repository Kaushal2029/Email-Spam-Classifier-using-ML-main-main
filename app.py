import warnings
from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import re

warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

app = Flask(__name__)

nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove short words
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

# Load data from CSV and train model
data = pd.read_csv("static/Spam Email Detection - spam.csv", usecols=[0, 1], names=["v1", "v2"], header=None)
data.dropna(inplace=True)  # Remove missing values
data.drop_duplicates(inplace=True)  # Remove duplicate entries

# Check for data imbalances and inspect the dataset
print("Class Distribution:", data['v1'].value_counts())

# Apply text preprocessing to the dataset
data['v2'] = data['v2'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['v2'], data['v1'], test_size=0.2, random_state=42)

# Model Training
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
clf = MultinomialNB()
clf.fit(X_train_tfidf, y_train)

# Model Evaluation (Only print in development)
X_test_tfidf = tfidf.transform(X_test)
y_pred = clf.predict(X_test_tfidf)
print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred))  # You can remove this in production

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        email = request.form.get('email-content')
        
        # Preprocess the input email
        preprocessed_email = preprocess_text(email)
        
        # Debugging: Print the preprocessed email
        print(f"Preprocessed Email: {preprocessed_email}")
        
        # Transform input with the trained TFIDF vectorizer
        tokenized_email = tfidf.transform([preprocessed_email])
        
        # Predict the class (Spam or Not Spam)
        prediction = clf.predict(tokenized_email)
        prediction = "Spam" if prediction[0] == 'spam' else "Not a Spam"
        
        # Debugging: Print the prediction
        print(f"Prediction: {prediction}")
        
    except Exception as e:
        prediction = f"An error occurred: {str(e)}"
    return render_template("index.html", prediction=prediction, email=email)

@app.route("/reset", methods=["GET"])
def reset():
    return render_template("index.html", prediction="", email="", other_field="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
