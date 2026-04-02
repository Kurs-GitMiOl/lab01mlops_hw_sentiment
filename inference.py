from sentence_transformers import SentenceTransformer
import joblib
import os


def load_models():
    # Absolute path to the model
    base_path = os.path.join(os.path.dirname(__file__), "models")

    # Paths to transformerr and clasyficer
    transformer_path = os.path.join(base_path, "sentence_transformer.model")
    classifier_path = os.path.join(base_path, "classifier.joblib")

    # Load entence transformer model
    transformer = SentenceTransformer(transformer_path)

    # Load regression clasificer
    classifier = joblib.load(classifier_path)

    return transformer, classifier


def predict(text: str, transformer, classifier) -> str:
    embedding = transformer.encode([text])

    # Predict sentiment class
    prediction = classifier.predict(embedding)[0]

    # Maping prediction
    mapping = {0: "negative", 1: "neutral", 2: "positive"}

    return mapping[prediction]
