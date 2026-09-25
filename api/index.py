import joblib
import re
from http.server import BaseHTTPRequestHandler
import json

model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

ENGLISH_STOP_WORDS = frozenset([
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","as","at","be","because","been","before","being","below","between",
    "both","but","by","could","did","do","does","doing","down","during","each",
    "few","for","from","further","had","has","have","having","he","her","here",
    "hers","herself","him","himself","his","how","i","if","in","into","is","it",
    "its","itself","just","me","more","most","my","myself","no","nor","not",
    "now","of","off","on","once","only","or","other","our","ours","ourselves",
    "out","over","own","same","she","should","so","some","such","than","that",
    "the","their","theirs","them","themselves","then","there","these","they",
    "this","those","through","to","too","under","until","up","very","was",
    "we","were","what","when","where","which","while","who","whom","why",
    "will","with","you","your","yours","yourself","yourselves"
])

def clean_text(text):
    text = text.lower()
    words = text.split()
    clean_words = [w for w in words if w.isalpha()]
    return ' '.join(clean_words)

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]
    return ' '.join(words)

def predict_message(message):
    cleaned = clean_text(message)
    cleaned = remove_stopwords(cleaned)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return "Spam" if prediction == 1 else "Ham"

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)
        text = data.get('text', '')

        result = predict_message(text)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        response = json.dumps({"message": text, "prediction": result})
        self.wfile.write(response.encode())
