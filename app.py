from pathlib import Path
import json
import re
import warnings

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
MODEL_PATH = PROJECT_ROOT / "saved_models" / "logistic_regression_tfidf.pkl"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "saved_models" / "logistic_tfidf_lr.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "saved_models" / "tfidf_vectorizer.joblib"
IMPROVED_MODEL_PATH = PROJECT_ROOT / "saved_models" / "improved_logistic_tfidf.pkl"
IMPROVED_METRICS_PATH = PROJECT_ROOT / "saved_models" / "improved_metrics.json"
CNN_MODEL_PATH = PROJECT_ROOT / "saved_models" / "cnn_fake_news_model.keras"
DL_TOKENIZER_PATH = PROJECT_ROOT / "saved_models" / "deep_learning_tokenizer.pkl"
FINAL_METADATA_PATH = PROJECT_ROOT / "saved_models" / "final_model_metadata.pkl"
DL_MAX_LENGTH = 300
CONFIDENCE_THRESHOLD = 0.65

LABELS = {
    0: "Real News",
    1: "Fake News",
}

URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)


st.set_page_config(
    page_title="Fake News Detection",
    layout="wide",
)


def basic_clean_text(text: str) -> str:
    """Basic cleaning from Fake_News_Detection_WELFake.ipynb."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def get_wordnet_lemmatizer():
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("tests")
        return lemmatizer
    except Exception:
        return None


def lemmatize_text(text: str) -> str:
    """Lemmatization used before training the tuned TF-IDF model."""
    lemmatizer = get_wordnet_lemmatizer()
    if lemmatizer is None:
        return text
    return " ".join(lemmatizer.lemmatize(token) for token in text.split())


def clean_for_tuned_model(text: str) -> str:
    return lemmatize_text(basic_clean_text(text))


def clean_for_report_model(text: str) -> str:
    """Cleaning from Fake_News_Project_Report.ipynb for the separate TF-IDF model."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9]", " ", text)
    text = re.sub(r"\+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_improved_metrics():
    if not IMPROVED_METRICS_PATH.exists():
        return {}
    try:
        with IMPROVED_METRICS_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception:
        return {}
    return {
        model.get("model"): model
        for model in data.get("models", [])
        if model.get("model")
    }


def get_improved_metric(model_name, metric_name, default=None):
    value = get_improved_metrics().get(model_name, {}).get(metric_name)
    if value is None:
        return default
    if metric_name in {"accuracy", "precision", "recall", "f1_score"}:
        return f"{value:.4f}"
    return value


@st.cache_resource
def load_models():
    models = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if CNN_MODEL_PATH.exists() and DL_TOKENIZER_PATH.exists():
            try:
                from tensorflow.keras.models import load_model
                from tensorflow.keras.preprocessing.sequence import pad_sequences

                models["1D CNN + Embedding"] = {
                    "kind": "keras_binary",
                    "model": load_model(CNN_MODEL_PATH),
                    "tokenizer": joblib.load(DL_TOKENIZER_PATH),
                    "pad_sequences": pad_sequences,
                    "max_length": DL_MAX_LENGTH,
                    "name": "1D CNN + Embedding",
                    "preprocess": "tuned",
                    "reported_accuracy": "0.9553 validation accuracy",
                }
            except Exception:
                pass

        if IMPROVED_MODEL_PATH.exists():
            models["Improved Logistic Regression + TF-IDF"] = {
                "kind": "pipeline",
                "model": joblib.load(IMPROVED_MODEL_PATH),
                "name": "Improved Logistic Regression + TF-IDF",
                "preprocess": "report",
                "reported_accuracy": get_improved_metric(
                    "Improved Logistic Regression + TF-IDF", "accuracy", "new"
                ),
            }

        if MODEL_PATH.exists():
            models["Logistic Regression + TF-IDF"] = {
                "kind": "pipeline",
                "model": joblib.load(MODEL_PATH),
                "name": "Logistic Regression + TF-IDF",
                "preprocess": "tuned",
                "reported_accuracy": "0.9406 validation accuracy",
            }

        if FALLBACK_MODEL_PATH.exists() and VECTORIZER_PATH.exists():
            models["Logistic Regression + TF-IDF"] = {
                "kind": "separate",
                "model": joblib.load(FALLBACK_MODEL_PATH),
                "vectorizer": joblib.load(VECTORIZER_PATH),
                "name": "Logistic Regression + TF-IDF",
                "preprocess": "report",
                "reported_accuracy": "0.9265 validation accuracy",
            }

    if models:
        return models

    raise FileNotFoundError(
        "No saved model found. Expected saved_models/logistic_regression_tfidf.pkl "
        "or saved_models/cnn_fake_news_model.keras plus saved_models/deep_learning_tokenizer.pkl, "
        "or saved_models/logistic_tfidf_lr.joblib plus saved_models/tfidf_vectorizer.joblib."
    )


def clean_for_model(model_bundle, text: str) -> str:
    if model_bundle["preprocess"] == "tuned":
        return clean_for_tuned_model(text)
    return clean_for_report_model(text)


def predict_rows(model_bundle, texts):
    cleaned = [clean_for_model(model_bundle, text) for text in texts]
    model = model_bundle["model"]

    if model_bundle["kind"] == "pipeline":
        features = cleaned
        predictions = model.predict(features)
    elif model_bundle["kind"] == "separate":
        features = model_bundle["vectorizer"].transform(cleaned)
        predictions = model.predict(features)
    else:
        sequences = model_bundle["tokenizer"].texts_to_sequences(cleaned)
        features = model_bundle["pad_sequences"](
            sequences,
            maxlen=model_bundle["max_length"],
            padding="post",
            truncating="post",
        )
        fake_probabilities = model.predict(features, verbose=0).ravel()
        predictions = (fake_probabilities >= 0.5).astype(int)
        real_probabilities = 1 - fake_probabilities

    if model_bundle["kind"] == "keras_binary":
        pass
    elif hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        fake_probabilities = probabilities[:, list(model.classes_).index(1)]
        real_probabilities = probabilities[:, list(model.classes_).index(0)]
    else:
        fake_probabilities = [None] * len(predictions)
        real_probabilities = [None] * len(predictions)

    return pd.DataFrame(
        {
            "cleaned_text": cleaned,
            "prediction": predictions,
            "prediction_label": [LABELS.get(int(pred), str(pred)) for pred in predictions],
            "real_probability": real_probabilities,
            "fake_probability": fake_probabilities,
            "max_confidence": [
                max(real, fake) if real is not None and fake is not None else None
                for real, fake in zip(real_probabilities, fake_probabilities)
            ],
            "low_confidence": [
                max(real, fake) < CONFIDENCE_THRESHOLD
                if real is not None and fake is not None
                else True
                for real, fake in zip(real_probabilities, fake_probabilities)
            ],
        }
    )


def probability_bar(label, value):
    if value is None:
        st.caption(f"{label}: probability unavailable")
        return
    st.metric(label, f"{value:.1%}")
    st.progress(float(value))


def show_input_quality_warnings(raw_text: str, cleaned_text: str):
    word_count = len(cleaned_text.split())
    has_url = bool(URL_PATTERN.search(raw_text))

    if has_url and word_count < 30:
        st.warning(
            "This looks like a URL or mostly a URL. The model was trained on article title "
            "and body text, not web links. Paste the article content for a fair prediction."
        )
    elif word_count < 60:
        st.info(
            "Short inputs such as headlines can be unreliable. For a stronger result, paste "
            "the full article body or at least several paragraphs."
        )


def render_single_prediction(model_bundle):
    st.subheader("Single Article Prediction")

    title = st.text_input("Article title", placeholder="Enter the article headline")
    article = st.text_area(
        "Article text",
        height=220,
        placeholder="Paste the article content here",
    )

    col_left, col_right = st.columns([1, 2])
    with col_left:
        predict_clicked = st.button("Predict", type="primary", use_container_width=True)

    combined = f"{title} {article}".strip()
    if predict_clicked:
        if not combined:
            st.warning("Enter a title, article text, or both.")
            return

        result = predict_rows(model_bundle, [combined]).iloc[0]
        prediction_label = result["prediction_label"]
        fake_probability = result["fake_probability"]
        real_probability = result["real_probability"]
        low_confidence = bool(result["low_confidence"])

        show_input_quality_warnings(combined, result["cleaned_text"])

        if low_confidence:
            st.warning(f"Uncertain prediction: leans {prediction_label}")
        elif int(result["prediction"]) == 1:
            st.error(f"Prediction: {prediction_label}")
        else:
            st.success(f"Prediction: {prediction_label}")

        prob_col_1, prob_col_2 = st.columns(2)
        with prob_col_1:
            probability_bar("Real news confidence", real_probability)
        with prob_col_2:
            probability_bar("Fake news confidence", fake_probability)

        with st.expander("Show cleaned text used by the model"):
            st.write(result["cleaned_text"])

        st.caption(
            f"Predictions below {CONFIDENCE_THRESHOLD:.0%} confidence are marked uncertain. "
            "This model is a decision-support demo, not a source-verification system."
        )


def render_batch_prediction(model_bundle):
    st.subheader("Batch CSV Prediction")
    st.write("Upload a CSV with a `text` column. A `title` column is optional.")

    uploaded_file = st.file_uploader("CSV file", type=["csv"])
    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    if "text" not in df.columns:
        st.error("The CSV must contain a `text` column.")
        return

    title_series = df["title"].fillna("") if "title" in df.columns else pd.Series([""] * len(df))
    texts = (title_series.astype(str) + " " + df["text"].fillna("").astype(str)).str.strip()
    predictions = predict_rows(model_bundle, texts.tolist())

    output = pd.concat([df.reset_index(drop=True), predictions.drop(columns=["cleaned_text"])], axis=1)

    st.dataframe(output, use_container_width=True, hide_index=True)
    st.download_button(
        "Download predictions",
        data=output.to_csv(index=False).encode("utf-8"),
        file_name="fake_news_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_model_performance(model_bundle):
    st.subheader("Model Comparison")
    st.write(f"Loaded model: `{model_bundle['name']}`")

    metrics = pd.DataFrame(
        [
            {
                "Model": "1D CNN + Embedding",
                "Category": "Deep Learning",
                "Feature Method": "Word embeddings + Conv1D",
                "Accuracy": 0.955285,
                "F1-score": 0.956148,
                "Evaluation": "Validation",
                "Strength": "Best validation F1-score; captures local phrase patterns",
                "Limitation": "Requires TensorFlow/Keras for deployment",
            },
            {
                "Model": "LSTM + Embedding",
                "Category": "Deep Learning",
                "Feature Method": "Word embeddings + sequence model",
                "Accuracy": 0.945591,
                "F1-score": 0.946626,
                "Evaluation": "Validation",
                "Strength": "Learns word-order patterns",
                "Limitation": "Slower and slightly weaker than CNN here",
            },
            {
                "Model": "Tuned Logistic Regression + TF-IDF",
                "Category": "Traditional ML",
                "Feature Method": "TF-IDF N-grams",
                "Accuracy": 0.940588,
                "F1-score": 0.941682,
                "Evaluation": "Validation",
                "Strength": "Strongest traditional ML model; explainable",
                "Limitation": "Does not model word order deeply",
            },
            {
                "Model": "Logistic Regression + TF-IDF N-grams",
                "Category": "Traditional ML",
                "Feature Method": "TF-IDF N-grams",
                "Accuracy": 0.926517,
                "F1-score": 0.928157,
                "Evaluation": "Validation",
                "Strength": "Strong baseline",
                "Limitation": "Weaker than tuned LR and deep learning models",
            },
            {
                "Model": "Naive Bayes + Bag of N-grams",
                "Category": "Traditional ML",
                "Feature Method": "Bag of N-grams",
                "Accuracy": 0.894934,
                "F1-score": 0.896488,
                "Evaluation": "Validation",
                "Strength": "Simple and fast baseline",
                "Limitation": "Weakest validation result",
            },
        ]
    )

    display_metrics = metrics.copy()
    display_metrics["Accuracy"] = display_metrics["Accuracy"].map(
        lambda value: "Not recorded" if pd.isna(value) else f"{value:.2%}"
    )
    display_metrics["F1-score"] = display_metrics["F1-score"].map(
        lambda value: "Not recorded" if pd.isna(value) else f"{value:.2%}"
    )

    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    chart_data = metrics.dropna(subset=["Accuracy"]).set_index("Model")["Accuracy"]
    st.bar_chart(chart_data)

    best_model = metrics.dropna(subset=["Accuracy"]).sort_values("Accuracy", ascending=False).iloc[0]
    deployable_model = "1D CNN + Embedding"

    col_best, col_deploy = st.columns(2)
    with col_best:
        st.metric("Best Accuracy", best_model["Model"], f"{best_model['Accuracy']:.2%}")
    with col_deploy:
        st.metric("Final Selected Model", deployable_model, "Highest validation F1-score")

    st.markdown(
        """
        **Presentation interpretation**

        The 1D CNN achieved the highest validation F1-score, so it was selected as the
        final model for testing and app prediction when TensorFlow is available. Tuned
        Logistic Regression remains the strongest traditional ML model and is useful as
        a fast, explainable fallback.
        """
    )


def render_about():
    st.subheader("About This Project")
    st.write(
        "This app demonstrates an NLP fake news classifier trained on the WELFake dataset. "
        "The project workflow includes text cleaning, TF-IDF representation, logistic regression, "
        "deep learning comparison, evaluation, and inference."
    )

    st.markdown(
        """
        **Label mapping**

        - `0`: Real News
        - `1`: Fake News

        **Demo features**

        - Predict one article from title and text
        - Show model confidence scores
        - Show cleaned text used for prediction
        - Upload a CSV and download batch predictions
        - Compare traditional ML and deep learning models
        """
    )


def main():
    st.title("Fake News Detection App")
    st.caption("WELFake NLP classifier demo")

    try:
        models = load_models()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    preferred_order = [
        "1D CNN + Embedding",
        "Tuned Logistic Regression + TF-IDF",
        "Logistic Regression + TF-IDF",
        "Improved Logistic Regression + TF-IDF",
    ]
    model_name = next((name for name in preferred_order if name in models), next(iter(models)))
    model_bundle = models[model_name]
    st.sidebar.subheader("Active Model")
    st.sidebar.write(model_bundle["name"])
    st.sidebar.caption(
        f"Reported score: {model_bundle['reported_accuracy']} | "
        f"Preprocessing: {model_bundle['preprocess']}"
    )

    tab_predict, tab_batch, tab_metrics, tab_about = st.tabs(
        ["Predict", "Batch Upload", "Model Comparison", "About"]
    )

    with tab_predict:
        render_single_prediction(model_bundle)

    with tab_batch:
        render_batch_prediction(model_bundle)

    with tab_metrics:
        render_model_performance(model_bundle)

    with tab_about:
        render_about()


if __name__ == "__main__":
    main()
