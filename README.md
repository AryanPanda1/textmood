# Textmood

Classify text sentiment with scikit-learn. Supports logistic regression, naive bayes, and random forest.

## Install
```bash
pip install numpy pandas scikit-learn
```

## Usage
```python
from sentiment_analyzer import SentimentAnalyzer

# train a model
analyzer = SentimentAnalyzer(model_type='logistic')
analyzer.train(X_train, y_train)

# make predictions
analyzer.predict("This is great!")  # 'positive'
analyzer.predict_proba("Not sure about this")  # array of probabilities

# save for later
analyzer.save('my_model.pkl')
```

## What it does

- Cleans up text (removes URLs, mentions, special characters)
- Turns text into features using TF-IDF
- Trains one of three classifiers
- Gives you predictions with confidence scores

## Example
```python
from sentiment_analyzer import create_sample_dataset

# get some sample data
texts, labels = create_sample_dataset()

# train
analyzer = SentimentAnalyzer()
analyzer.train(texts, labels)

# evaluate
metrics = analyzer.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']}")
```

## Model options

- `logistic` - Logistic Regression (default, usually works well)
- `naive_bayes` - Naive Bayes (fast, good for large datasets)
- `random_forest` - Random Forest (slower but sometimes more accurate)

## Run the demo
```bash
python sentiment_analyzer.py
```

This trains a model on sample data and shows predictions.

## Notes

The preprocessing is pretty aggressive - it lowercase everything and strips out URLs, mentions, and non-alphabetic characters. If you need something different, just modify the `preprocess_text` method.

## License

MIT
