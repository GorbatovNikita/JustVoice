import json
import pickle
import logging
import re
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

TRAINING_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "train.json"
MODEL_PATH = Path(__file__).parent.parent.parent / "data" / "topic_classifier.pkl"
MODEL_PATH.parent.mkdir(exist_ok=True)


class TopicClassifier:
    
    def __init__(self):
        self.pipeline = None
        self.topics = []
        self.is_fitted = False
        self.training_data = None
        self.label_encoder = LabelEncoder()
    
    def _load_training_data(self):
        if self.training_data is not None:
            return self.training_data
        
        with open(TRAINING_DATA_PATH, 'r', encoding='utf-8') as f:
            self.training_data = json.load(f)
        
        return self.training_data
    
    def _preprocess_text(self, text):
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _prepare_data(self):
        data = self._load_training_data()
        texts = []
        labels = []
        
        for topic, lang_data in data["topics"].items():
            if topic not in self.topics:
                self.topics.append(topic)
            
            for lang, examples in lang_data.items():
                if isinstance(examples, list):
                    if all(isinstance(ex, str) and len(ex.split()) > 1 for ex in examples[:3]):
                        for example in examples:
                            texts.append(self._preprocess_text(example))
                            labels.append(topic)
                    else:
                        texts.append(" ".join(examples))
                        labels.append(topic)
        
        return texts, labels
    
    def _augment_data(self, texts, labels):
        augmented_texts = []
        augmented_labels = []
        
        for text, label in zip(texts, labels):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            words = text.split()
            if len(words) >= 3:
                for _ in range(5):
                    n_words = np.random.randint(2, min(len(words) + 1, 8))
                    subset = np.random.choice(words, size=n_words, replace=False)
                    augmented_texts.append(" ".join(subset))
                    augmented_labels.append(label)
        
        return augmented_texts, augmented_labels
    
    def train(self):
        logger.info("Training topic classifier...")
        
        texts, labels = self._prepare_data()
        texts, labels = self._augment_data(texts, labels)
        
        logger.info(f"Training on {len(texts)} samples across {len(self.topics)} topics")
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='word',
                ngram_range=(1, 3),
                max_features=10000,
                min_df=1,
                max_df=0.95,
                lowercase=True,
                strip_accents='unicode',
                sublinear_tf=True,
                use_idf=True,
                norm='l2'
            )),
            ('clf', LinearSVC(
                C=0.5,
                class_weight='balanced',
                max_iter=5000,
                random_state=42,
                dual=True,
                tol=1e-4
            ))
        ])
        
        self.pipeline.fit(texts, labels)
        self.is_fitted = True
        
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump({
                'pipeline': self.pipeline,
                'topics': self.topics
            }, f)
        
        logger.info("Model trained and saved")
    
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
        
        original_text = text
        text = self._preprocess_text(text)
        
        data = self._load_training_data()
        
        keyword_scores = {}
        for topic in self.topics:
            score = 0
            for lang_data in data["topics"][topic].values():
                if isinstance(lang_data, list):
                    for item in lang_data:
                        if isinstance(item, str):
                            if len(item.split()) <= 3:
                                score += len(re.findall(r'\b' + re.escape(item) + r'\b', text))
                            else:
                                if item.lower() in text:
                                    score += 3
            keyword_scores[topic] = score
        
        max_score = max(keyword_scores.values()) if keyword_scores else 0
        
        if max_score >= 2:
            best_topic = max(keyword_scores, key=keyword_scores.get)
            logger.info(f"Keyword match: {best_topic} (score: {max_score})")
            return best_topic
        
        try:
            probabilities = self.pipeline.decision_function([text])[0]
            if isinstance(probabilities, np.ndarray) and len(probabilities) > 0:
                best_idx = np.argmax(probabilities)
                confidence = probabilities[best_idx]
                
                if confidence > 0.1:
                    logger.info(f"ML prediction: {self.topics[best_idx]} (confidence: {confidence:.2f})")
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
    _topic_classifier = TopicClassifier()
    _topic_classifier.train()