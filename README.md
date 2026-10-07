# SpamShield — Spam Classifier Website

A web interface for the final spam-classification approach from the notebook:

**NLP preprocessing → TF-IDF (`max_features=3000`) → Multinomial Naive Bayes**

# Live Website Demo 

https://spam-email-calssifier.onrender.com/

The notebook's reported test result for this configuration was approximately:

- Accuracy: **97.10%**
- Precision: **100%**

## Project structure

```text
spam_classifier_website/
├── index.html
├── style.css
├── script.js
├── app.py
├── train_model.py
├── requirements.txt
└── spam.csv              # add your dataset here
```

## Setup

Open a terminal in this folder:

```bash
pip install -r requirements.txt
```

Put the **same `spam.csv` used for the notebook** in this folder.

Then train/save the model:

```bash
python train_model.py
```

This creates:

```text
spam_model.pkl
```

Start the website:

```bash
python app.py
```

Open the local address printed by Flask, normally:

```text
http://127.0.0.1:5000
```

## Why there is a Python backend

The trained model uses scikit-learn's `TfidfVectorizer` and `MultinomialNB`. A plain HTML/CSS/JS file cannot directly execute that Python/scikit-learn model.

The browser frontend sends the message to the Flask `/predict` endpoint, and the Python backend runs the actual trained pipeline.
