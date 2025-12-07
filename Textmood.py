# sentiment_analyzer.py
"""
Sentiment Analysis Tool
A machine learning project for analyzing sentiment in text using multiple models.
"""

import re
import pickle
from typing import List, Union, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


class SentimentAnalyzer:
    """
    A sentiment analysis classifier that supports multiple ML algorithms.

    Attributes:
        model_type (str): Type of model to use ('logistic', 'naive_bayes', 'random_forest')
        vectorizer (TfidfVectorizer): Text vectorizer
        model: Trained ML model
    """

    def __init__(self, model_type: str = 'logistic') -> None:
        """
        Initialize the sentiment analyzer.

        Args:
            model_type (str): Model type - 'logistic', 'naive_bayes', or 'random_forest'
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = self._initialize_model()

    def _initialize_model(self):
        """Initialize the selected model."""
        models = {
            'logistic': LogisticRegression(max_iter=1000, random_state=42),
            'naive_bayes': MultinomialNB(),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        }
        return models.get(self.model_type, models['logistic'])

    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text data.

        Args:
            text (str): Raw text input

        Returns:
            str: Cleaned text
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text

    def train(self, X_train: List[str], y_train: List[str]) -> None:
        """
        Train the sentiment analysis model.

        Args:
            X_train (list): Training texts
            y_train (list): Training labels
        """
        X_train_clean = [self.preprocess_text(text) for text in X_train]
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        self.model.fit(X_train_vec, y_train)
        print(f"Model trained successfully using {self.model_type}")

    def predict(self, texts: Union[str, List[str]]):
        """
        Predict sentiment for given texts.

        Args:
            texts (list or str): Text(s) to analyze

        Returns:
            numpy.ndarray or str: Predicted sentiment(s)
        """
        single = False
        if isinstance(texts, str):
            texts = [texts]
            single = True

        texts_clean = [self.preprocess_text(text) for text in texts]
        texts_vec = self.vectorizer.transform(texts_clean)
        predictions = self.model.predict(texts_vec)

        return predictions[0] if single else predictions

    def predict_proba(self, texts: Union[str, List[str]]):
        """
        Get probability predictions.

        Args:
            texts (list or str): Text(s) to analyze

        Returns:
            numpy.ndarray: Probability predictions
        """
        if isinstance(texts, str):
            texts = [texts]

        texts_clean = [self.preprocess_text(text) for text in texts]
        texts_vec = self.vectorizer.transform(texts_clean)
        return self.model.predict_proba(texts_vec)

    def evaluate(self, X_test: List[str], y_test: List[str]) -> Dict[str, Any]:
        """
        Evaluate model performance.

        Args:
            X_test (list): Test texts
            y_test (list): Test labels

        Returns:
            dict: Evaluation metrics
        """
        predictions = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'classification_report': classification_report(y_test, predictions),
            'confusion_matrix': confusion_matrix(y_test, predictions),
        }

    def save(self, filepath: str = 'sentiment_model.pkl') -> None:
        """Save model and vectorizer to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({'model': self.model, 'vectorizer': self.vectorizer}, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str = 'sentiment_model.pkl') -> None:
        """Load model and vectorizer from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        print(f"Model loaded from {filepath}")


def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is the worst experience I've ever had. Completely disappointed.",
        "Pretty good overall, though there's room for improvement.",
        "Absolutely terrible service. Would not recommend to anyone.",
        "Fantastic! Exceeded all my expectations.",
        "Not bad, but nothing special either.",
        "Horrible quality. Complete waste of money.",
        "I'm very satisfied with this purchase. Great value!",
        "Mediocre at best. Expected much more for the price.",
        "Outstanding! This is exactly what I was looking for.",
    ]

    labels = [
        'positive', 'negative', 'neutral', 'negative', 'positive',
        'neutral', 'negative', 'positive', 'neutral', 'positive',
    ]

    return texts, labels


def main() -> None:
    """Main function to demonstrate the sentiment analyzer."""
    print("=== Sentiment Analysis Demo ===\n")

    texts, labels = create_sample_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )

    analyzer = SentimentAnalyzer(model_type='logistic')
    analyzer.train(X_train, y_train)

    print("\n=== Model Evaluation ===")
    metrics = analyzer.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.2f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    print("\n=== Sample Predictions ===")
    test_samples = [
        "This is absolutely wonderful!",
        "I'm very unhappy with this.",
        "It's okay, nothing special.",
    ]

    for text in test_samples:
        prediction = analyzer.predict(text)
        proba = analyzer.predict_proba(text)
        print(f"\nText: '{text}'")
        print(f"Sentiment: {prediction}")
        print(f"Confidence: {max(proba[0]):.2%}")

    analyzer.save('sentiment_model.pkl')


if __name__ == "__main__":
    main()
