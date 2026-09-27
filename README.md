# 📩 SMS Spam Classifier

An end-to-end machine learning project that classifies SMS messages as **Spam** or **Ham** (not spam), deployed as a live, publicly callable API.

**🔗 Live API:** `https://spam-classifier-p9a5pt369-inno-93f5.vercel.app/api/index`

---

## 📊 Project Overview

This project walks through the complete ML workflow — from raw text data to a deployed, production-style API:

1. **Exploratory Data Analysis (EDA)**
2. **Text Preprocessing**
3. **Model Training & Comparison**
4. **Deployment**

Dataset: [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,157 labeled messages (4,516 ham / 641 spam).

---

## 🔍 Key EDA Findings

- **Spam vocabulary** is dominated by promotional/urgency words: `free`, `call`, `claim`, `urgent`, `txt`, `mobile`, `reply`, `stop`
- **Ham vocabulary** reflects casual conversation: `will`, `come`, `know`, `home`, `love`, `good`
- **`digit_count`** showed a strong correlation (**0.84**) with spam — spam messages contain far more numbers (phone numbers, prices, codes) than ham

---

## 🤖 Model Comparison

Four approaches were trained and evaluated on the same test set:

| Model | Accuracy | False Positives (Ham→Spam) | False Negatives (Missed Spam) | Spam Precision | Spam Recall |
|---|---|---|---|---|---|
| CountVectorizer + Naive Bayes | 94.9% | 41 | 12 | 0.75 | 0.91 |
| SMOTE-balanced + Naive Bayes | 91.6% | 77 | 10 | 0.62 | 0.93 |
| **TF-IDF + Naive Bayes** ⭐ | **95.7%** | **1** | 43 | **0.99** | 0.68 |
| TF-IDF + Logistic Regression | 93.5% | 5 | 62 | 0.94 | 0.54 |

**Chosen model: TF-IDF + Naive Bayes** — best overall accuracy and near-perfect precision, meaning it almost never flags a legitimate message as spam. Trade-off: it's more conservative about catching every spam message, which is an acceptable trade-off for most real-world use cases where false alarms are more disruptive than an occasional missed spam message.

---

## ⚙️ Preprocessing Pipeline

1. Lowercase all text
2. Remove numbers and punctuation
3. Remove common stopwords
4. Vectorize with **TF-IDF** (Term Frequency–Inverse Document Frequency)

---

## 🚀 API Usage

**Endpoint:** `POST /api/index`

**Request:**
```json
{
  "text": "Congratulations! You've won a free prize, call now"
}
```

**Response:**
```json
{
  "message": "Congratulations! You've won a free prize, call now",
  "prediction": "Spam"
}
```

**Example (Python):**
```python
import requests

response = requests.post(
    "https://spam-classifier-p9a5pt369-inno-93f5.vercel.app/api/index",
    json={"text": "Hey, are we still meeting for lunch today?"}
)
print(response.json())
# {'message': 'Hey, are we still meeting for lunch today?', 'prediction': 'Ham'}
```

---

## 🛠️ Tech Stack

- **Python** — pandas, scikit-learn
- **Modeling** — Multinomial Naive Bayes, Logistic Regression, TF-IDF, SMOTE
- **Deployment** — Vercel (serverless functions)
- **Version control** — GitHub

---

## 📁 Repository Structure

```
spam-classifier-api/
├── api/
│   └── index.py          # Serverless API function
├── spam_model.pkl         # Trained TF-IDF + Naive Bayes model
├── tfidf_vectorizer.pkl   # Fitted TF-IDF vectorizer
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 📈 Possible Future Improvements

- Try additional algorithms (SVM, ensemble methods)
- Expand deployment with authentication for production use
- Add a simple front-end for non-technical users to test messages
