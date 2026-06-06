# Fake News Detection Streamlit App

This folder contains a Streamlit demo for the trained WELFake fake news classifier.

## Run

From the project root:

```bash
streamlit run app/app.py
```

The app loads `saved_models/tuned_logistic_regression_tfidf.pkl` by default. If that file is missing, it falls back to `saved_models/logistic_tfidf_lr.joblib` and `saved_models/tfidf_vectorizer.joblib`.

## Features

- Single article prediction from title and text
- Confidence scores for Real News and Fake News
- Cleaned model input preview
- CSV batch upload and downloadable predictions
- Model performance summary for presentation

CSV batch uploads should include a `text` column. A `title` column is optional.
