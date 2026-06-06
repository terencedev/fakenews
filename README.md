# Fake News Detection Streamlit App

This folder contains a Streamlit demo for the trained WELFake fake news classifier.

## Run

From the project root:

```bash
streamlit run app.py
```

The app loads `saved_models/cnn_fake_news_model.keras` and `saved_models/deep_learning_tokenizer.pkl` when TensorFlow is installed. For Streamlit deployment, the required dependency set uses the smaller scikit-learn fallback model at `saved_models/logistic_regression_tfidf.pkl`.

For local CNN inference, install TensorFlow separately:

```bash
pip install tensorflow
```

## Features

- Single article prediction from title and text
- Confidence scores for Real News and Fake News
- Cleaned model input preview
- CSV batch upload and downloadable predictions
- Model performance summary for presentation

CSV batch uploads should include a `text` column. A `title` column is optional.
