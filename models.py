# models.py
from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter
import random

random.seed(44)
np.random.seed(44)

class FeatureExtractor(object):
    """Base class for feature extraction."""
    def get_indexer(self):
        raise Exception("Call subclass implementation.")

    def extract_features(self, sentence: list, add_to_indexer: bool = False) -> Counter:
        raise Exception("Call subclass implementation.")

class UnigramFeatureExtractor(FeatureExtractor):
    """Extracts unigram features."""
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def fit_and_extract_features(self, tokenized_sentences: list):
        processed = [preprocess_text(s) for s in tokenized_sentences]
        all_tokens = [token for sent in processed for token in sent]
        freq = Counter(all_tokens)
        n = len(processed)
        self.pruned = {w: c for w, c in freq.items() if c > 1 and c <= n * 0.2}
        features = []
        for sent in processed:
            indices = [self.indexer.add_and_get_index(tok) for tok in sent if tok in self.pruned]
            features.append(indices)
        self.max_id = max(self.indexer.ints_to_objs.keys())
        mat = np.zeros((len(processed), self.max_id))
        for i, idxs in enumerate(features):
            for idx in idxs:
                mat[i][idx - 1] += 1
        return mat
    def extract_features(self, sentence: list, add_to_indexer: bool = False) -> Counter:
        sent = preprocess_text(sentence)
        filtered = [tok for tok in sent if tok in self.pruned]
        idxs = [self.indexer.index_of(tok) for tok in filtered]
        mat = np.zeros((1, self.max_id))
        for idx in idxs:
            mat[0, idx - 1] += 1
        return mat

class BigramFeatureExtractor(FeatureExtractor):
    """Extracts bigram features."""
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def fit_and_extract_features(self, tokenized_sentences: list):
        processed = [preprocess_text(s) for s in tokenized_sentences]
        bigrammed = [add_bigrams(s) for s in processed]
        all_tokens = [token for sent in bigrammed for token in sent]
        freq = Counter(all_tokens)
        n = len(bigrammed)
        self.pruned = {w: c for w, c in freq.items() if c > 1 and c <= n * 0.2}
        features = []
        for sent in bigrammed:
            indices = [self.indexer.add_and_get_index(tok) for tok in sent if tok in self.pruned]
            features.append(indices)
        self.max_id = max(self.indexer.ints_to_objs.keys())
        mat = np.zeros((len(bigrammed), self.max_id))
        for i, idxs in enumerate(features):
            for idx in idxs:
                mat[i][idx - 1] += 1
        return mat

    def extract_features(self, sentence: list, add_to_indexer: bool = False) -> Counter:
        sent = preprocess_text(sentence)
        bigrammed = add_bigrams(sent)
        filtered = [tok for tok in bigrammed if tok in self.pruned]
        idxs = [self.indexer.index_of(tok) for tok in filtered]
        mat = np.zeros((1, self.max_id))
        for idx in idxs:
            mat[0, idx - 1] += 1
        return mat

class BetterFeatureExtractor(FeatureExtractor):
    """Advanced feature extractor."""
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def fit_and_extract_features(self, tokenized_sentences: list):
        processed = [preprocess_text(s) for s in tokenized_sentences]
        all_tokens = [token for sent in processed for token in sent]
        freq = Counter(all_tokens)
        n = len(processed)
        self.pruned = {w: c for w, c in freq.items() if c > 1 and c <= n * 0.3}
        features = []
        for sent in processed:
            indices = [self.indexer.add_and_get_index(tok) for tok in sent if tok in self.pruned]
            features.append(indices)
        self.max_id = max(self.indexer.ints_to_objs.keys())
        mat = np.zeros((len(processed), self.max_id))
        for i, idxs in enumerate(features):
            for idx in idxs:
                mat[i][idx - 1] += 1
        return mat

    def extract_features(self, sentence: list, add_to_indexer: bool = False) -> Counter:
        sent = preprocess_text(sentence)
        filtered = [tok for tok in sent if tok in self.pruned]
        idxs = [self.indexer.index_of(tok) for tok in filtered]
        mat = np.zeros((1, self.max_id))
        for idx in idxs:
            mat[0, idx - 1] += 1
        return mat

class SentimentClassifier(object):
    """Base sentiment classifier."""
    def predict(self, sentence: list) -> int:
        raise Exception("Call subclass implementation.")

class TrivialSentimentClassifier(SentimentClassifier):
    """Always predicts positive."""
    def predict(self, sentence: list) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    """Perceptron classifier."""
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def fit(self, train_exs):
        sentences = [ex.words for ex in train_exs]
        labels = [ex.label for ex in train_exs]
        X = self.feature_extractor.fit_and_extract_features(sentences)
        self.weights = np.zeros((1, X.shape[1]))
        for epoch in range(25):
            for i in range(X.shape[0]):
                pred = int(np.dot(X[i:i+1], self.weights.T)[0][0] >= 0)
                self.weights += (1e-8 / np.sqrt(epoch + 1)) * ((labels[i] - pred) * X[i:i+1])

    def predict(self, sentence: list) -> int:
        X = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        return int(np.dot(X, self.weights.T)[0][0] >= 0)

class LogisticRegressionClassifier(SentimentClassifier):
    """Logistic regression classifier."""
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    def fit(self, train_exs, dev_exs):
        random.shuffle(train_exs)
        sentences = [ex.words for ex in train_exs]
        labels = [ex.label for ex in train_exs]
        X = self.feature_extractor.fit_and_extract_features(sentences)
        self.weights = np.random.normal(0, 0.4, (1, X.shape[1]))
        y_arr = np.array(labels).reshape(len(labels), 1)
        for epoch in range(25):
            for i in range(len(X)):
                X_i = X[i:i+1]
                y_i = y_arr[i:i+1]
                score = np.dot(X_i, self.weights.T)
                prob = np.exp(score) / (1 + np.exp(score))
                diff = y_i - prob
                update = np.dot(diff.T, X_i)
                self.weights += (0.2 / np.sqrt(epoch + 2)) * update

    def predict(self, sentence: list) -> int:
        X = self.feature_extractor.extract_features(sentence, add_to_indexer=False)
        score = np.dot(X, self.weights.T)[0][0]
        prob = np.exp(score) / (1 + np.exp(score))
        return prob >= 0.5

def train_perceptron(train_exs: list, feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    model = PerceptronClassifier(feat_extractor)
    model.fit(train_exs)
    return model
def train_logistic_regression(train_exs: list, feat_extractor: FeatureExtractor, dev_exs) -> LogisticRegressionClassifier:
    model = LogisticRegressionClassifier(feat_extractor)
    model.fit(train_exs, dev_exs)
    return model

def add_bigrams(sentence: list):
    import copy
    shifted = ["BEGIN"] + copy.deepcopy(sentence)
    bigrams = [sentence[i] + "_" + shifted[i] for i in range(len(sentence))]
    return sentence + bigrams

def preprocess_text(sentence: list):
    split = []
    for tok in sentence:
        if "-" in tok:
            split += tok.split("-")
        else:
            split.append(tok)
    return [tok.lower() for tok in split if tok.isalpha()]

def train_model(args, train_exs: list, dev_exs: list) -> SentimentClassifier:
    random.shuffle(train_exs)
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, dev_exs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
