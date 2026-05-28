import json
import pickle
import logging
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

TRAINING_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "train.json"
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "topic_classifier.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)
print(TRAINING_DATA_PATH)


class TopicClassifier:
    
    def __init__(self):
        self.pipeline = None
        self.topics = []
        self.is_fitted = False
        self.training_data = None
    
    def _load_training_data(self):
        if self.training_data is not None:
            return self.training_data
        
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            self.training_data = json.load(f)
        
        logger.info(f"Training data loaded from {TRAINING_DATA_PATH}")
        return self.training_data
    
    def _prepare_data(self):
        data = self._load_training_data()
        texts = []
        labels = []
        
        for topic, lang_data in data["topics"].items():
            if topic not in self.topics:
                self.topics.append(topic)
            
            for lang, words in lang_data.items():
                texts.append(" ".join(words))
                labels.append(topic)
        
        return texts, labels
    
    def _augment_data(self, texts, labels):
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            words = text.split()
            for _ in range(10):
                if len(words) >= 2:
                    n_words = np.random.randint(2, max(3, len(words) + 1))
                    subset = np.random.choice(words, size=min(n_words, len(words)), replace=False)
                    augmented_texts.append(" ".join(subset))
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def train(self):
        texts, labels = self._prepare_data()
        texts, labels = self._augment_data(texts, labels)
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 2),
                max_features=8000,
                min_df=1,
                lowercase=True,
                strip_accents='unicode'
            )),
            ('clf', LinearSVC(
                C=0.3,
                class_weight='balanced',
                max_iter=3000,
                random_state=42,
                dual=True
            ))
        ])
        
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'topics': self.topics
            }, f)
        
    
    def load(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                data = pickle.load(f)
                self.pipeline = data['pipeline']
                self.topics = data['topics']
                self.is_fitted = True
            return True
        return False
    
    def predict(self, text):
        if not self.is_fitted:
            self.train()
        
        if not text or len(text.strip()) < 10:
            return "general"
        
        text = text.lower().strip()
        
        data = self._load_training_data()
        
        keyword_scores = {}
        for topic in self.topics:
            score = 0
            for lang_data in data["topics"][topic].values():
                score += sum(1 for word in lang_data if word in text)
            keyword_scores[topic] = score
        
        best_keyword_topic = max(keyword_scores, key=keyword_scores.get)
        if keyword_scores[best_keyword_topic] >= 3:
            return best_keyword_topic
        
        try:
            probabilities = self.pipeline.decision_function([text])[0]
            if isinstance(probabilities, np.ndarray) and len(probabilities) > 0:
                best_idx = np.argmax(probabilities)
                return self.topics[best_idx]
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
        
        return "general"


_topic_classifier = None


def get_topic_classifier():
    global _topic_classifier
    if _topic_classifier is None:
        _topic_classifier = TopicClassifier()
        if not _topic_classifier.load():
            _topic_classifier.train()
    return _topic_classifier


def classify_topic(text):
    classifier = get_topic_classifier()
    return classifier.predict(text)


def retrain_model():
    global _topic_classifier
    logger.info("Retraining topic classifier...")
    _topic_classifier = TopicClassifier()
    _topic_classifier.train()
    logger.info("Retraining complete")