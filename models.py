# models.py
from sentiment_data import *
from utils import *
import numpy as np
from collections import Counter
# from sentiment_classifier import evaluate
import random
random.seed(44)
np.random.seed(44)

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def fit_and_extract_features(self, tokenized_sentences: List[List[str]]):
        extracted_features = []
        tokenized_sentences = [preprocess_text(sentence) for sentence in tokenized_sentences]

        # function for Word counting and Tokenizetion
        import itertools
        dict_with_word_counts = Counter(itertools.chain(*tokenized_sentences))
        num_examples = len(tokenized_sentences)
        self.dict_with_word_counts_pruned = {key: val for key, val in dict_with_word_counts.items() if (val > 1 and val <= (num_examples * 0.20))}
        
        for tokenized_sentence in tokenized_sentences:
            feat = []
            for tok in tokenized_sentence:
                if tok in self.dict_with_word_counts_pruned.keys():
                    feat.append(self.indexer.add_and_get_index(tok))
            extracted_features.append(feat)
        
        self.max_id = get_max_index(self.indexer)
        feature_matrix = np.zeros((len(tokenized_sentences), self.max_id))
        
        for i, sentence_indices in enumerate(extracted_features):
            for tok_id in sentence_indices:
                feature_matrix[i][tok_id - 1] = feature_matrix[i][tok_id - 1] + 1
        # feature_matrix[i][tok_id - 1] = 1
        # print(f"Number of features = {self.max_id}")
        return feature_matrix
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        sentence = preprocess_text(sentence)
        sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        pruned_sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        ls_feats = [self.indexer.index_of(tok) for tok in pruned_sentence]
        feature_matrix = np.zeros((1, self.max_id))
        
        for tok_id in ls_feats:
            feature_matrix[0, tok_id - 1] = feature_matrix[0, tok_id - 1] + 1
        # feature_matrix[0, tok_id - 1] = 1
        return feature_matrix

def get_max_index(indexer):
    return max(indexer.ints_to_objs.keys())

class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def fit_and_extract_features(self, tokenized_sentences: List[List[str]]):
        extracted_features = []
        tokenized_sentences = [preprocess_text(sentence) for sentence in tokenized_sentences]
        tokenized_sentences = [add_bigrams(sentence) for sentence in tokenized_sentences]

        # Count frequency of words
        import itertools
        dict_with_word_counts = Counter(itertools.chain(*tokenized_sentences))
        num_examples = len(tokenized_sentences)
        self.dict_with_word_counts_pruned = {key: val for key, val in dict_with_word_counts.items() if (val > 1 and val <= (num_examples * 0.20))}
        
        for tokenized_sentence in tokenized_sentences:
            feat = []
            for tok in tokenized_sentence:
                if tok in self.dict_with_word_counts_pruned.keys():
                    feat.append(self.indexer.add_and_get_index(tok))
            extracted_features.append(feat)
        
        self.max_id = get_max_index(self.indexer)
        feature_matrix = np.zeros((len(tokenized_sentences), self.max_id))
        
        for i, sentence_indices in enumerate(extracted_features):
            for tok_id in sentence_indices:
                feature_matrix[i][tok_id - 1] = feature_matrix[i][tok_id - 1] + 1
        # Feature_matrix[i][tok_id - 1] = 1
        # Print(f"Number of features = {self.max_id}")
        return feature_matrix
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        sentence = preprocess_text(sentence)
        sentence = add_bigrams(sentence)
        sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        pruned_sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        ls_feats = [self.indexer.index_of(tok) for tok in pruned_sentence]
        feature_matrix = np.zeros((1, self.max_id))
        
        for tok_id in ls_feats:
            feature_matrix[0, tok_id - 1] = feature_matrix[0, tok_id - 1] + 1
        # Feature_matrix[0, tok_id - 1] = 1
        return feature_matrix

class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
    
    def fit_and_extract_features(self, tokenized_sentences: List[List[str]]):
        extracted_features = []
        tokenized_sentences = [preprocess_text(sentence) for sentence in tokenized_sentences]

        # Frequency of words count
        import itertools
        dict_with_word_counts = Counter(itertools.chain(*tokenized_sentences))
        num_examples = len(tokenized_sentences)
        self.dict_with_word_counts_pruned = {key: val for key, val in dict_with_word_counts.items() if (val > 1 and val <= (num_examples * 0.30))}
        
        for tokenized_sentence in tokenized_sentences:
            feat = []
            for tok in tokenized_sentence:
                if tok in self.dict_with_word_counts_pruned.keys():
                    feat.append(self.indexer.add_and_get_index(tok))
            extracted_features.append(feat)
        
        self.max_id = get_max_index(self.indexer)
        feature_matrix = np.zeros((len(tokenized_sentences), self.max_id))
        
        for i, sentence_indices in enumerate(extracted_features):
            for tok_id in sentence_indices:
                feature_matrix[i][tok_id - 1] = feature_matrix[i][tok_id - 1] + 1
        # Feature_matrix[i][tok_id - 1] = 1
        # Print(f"Number of features = {self.max_id}")
        return feature_matrix
    
    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        sentence = preprocess_text(sentence)
        sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        pruned_sentence = [tok for tok in sentence if tok in self.dict_with_word_counts_pruned.keys()]
        ls_feats = [self.indexer.index_of(tok) for tok in pruned_sentence]
        feature_matrix = np.zeros((1, self.max_id))
        
        for tok_id in ls_feats:
            feature_matrix[0, tok_id - 1] = feature_matrix[0, tok_id - 1] + 1
        # eature_matrix[0, tok_id - 1] = 1
        return feature_matrix

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """
    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """
    def predict(self, sentence: List[str]) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have __init__() and implement the
    predict method from the SentimentClassifier superclass. Hint: you'll probably need this class 
    to wrap both the weight vector and featurizer -- feel free to modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def fit(self, train_exs):
        """
        Args:
        X: sentences
        y: labels
        Returns:
        """
        sentences = [ex.words for ex in train_exs]
        y = [ex.label for ex in train_exs]
        X_train = self.feature_extractor.fit_and_extract_features(sentences)
        self.weights = np.zeros((1, X_train.shape[1]))
        
        for t in range(25):  # Epochs
            for i in range(X_train.shape[0]):  # Example
                y_pred = int(np.matmul(X_train[i:i + 1], self.weights.transpose())[0][0] >= 0)
                self.weights = self.weights + (1e-8 / np.sqrt(t + 1)) * ((y[i] - y_pred) * X_train[i:i + 1])
    
    def predict(self, sentence: List[str]) -> int:
        X_infer = self.feature_extractor.extract_features(sentence=sentence, add_to_indexer=False)
        return int(np.matmul(X_infer, self.weights.transpose())[0][0] >= 0)

class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have __init__() and implement the
    predict method from the SentimentClassifier superclass. Hint: you'll probably need this class 
    to wrap both the weight vector and featurizer -- feel free to modify the constructor to pass these in.
    """
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
    
    def fit(self, train_exs, dev_exs):
        """
        Args:
        train_exs
        Returns:
        """
        random.shuffle(train_exs)
        sentences = [ex.words for ex in train_exs]
        y = [ex.label for ex in train_exs]
        X_train = self.feature_extractor.fit_and_extract_features(sentences)
        self.weights = np.random.normal(0, 0.4, (1, X_train.shape[1]))
        y_arr = np.reshape(np.array(y), (len(y), 1))
        
        for t in range(25):
            for i in range(len(X_train)):
                X_small = X_train[i:i+1]
                y_small = y_arr[i:i+1]
                scores = np.matmul(X_small, self.weights.transpose())
                prob_positive = np.exp(scores) / (1 + np.exp(scores))
                diff = y_small - prob_positive
                update = np.matmul(diff.transpose(), X_small)
                self.weights = self.weights + (0.2/np.sqrt(t+2)) * update
    
    def predict(self, sentence: List[str]) -> int:
        X_infer = self.feature_extractor.extract_features(sentence=sentence, add_to_indexer=False)
        score = np.matmul(X_infer, self.weights.transpose())[0][0]
        prob_positive = np.exp(score) / (1 + np.exp(score))
        return prob_positive >= 0.50

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """
    perceptron = PerceptronClassifier(feat_extractor)
    perceptron.fit(train_exs)
    return perceptron

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, dev_exs) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """
    lr = LogisticRegressionClassifier(feat_extractor)
    lr.fit(train_exs, dev_exs)
    return lr

def add_bigrams(sentence: List[str]):
    import copy
    sentence_shift = ["BEGIN"] + copy.deepcopy(sentence)
    bigrams = []
    for i in range(len(sentence)):
        bigrams.append(sentence[i] + "_" + sentence_shift[i])
    return sentence + bigrams

def preprocess_text(sentence: List[str]):
    split_sentence = []
    for tok in sentence:
        if "-" in tok:
            split_sentence = split_sentence + tok.split("-")
        else:
            split_sentence.append(tok)
    new_sentence = []
    for tok in split_sentence:
        if tok.isalpha():
            new_sentence.append(tok.lower())
    return new_sentence

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Shuffle the train examples
    random.shuffle(train_exs)
    # Initialize feature extractor
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
    
    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, dev_exs)
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    
    return model


# # models.py

# from sentiment_data import *
# from utils import *

# from collections import Counter

# class FeatureExtractor(object):
#     """
#     Feature extraction base type. Takes a sentence and returns an indexed list of features.
#     """
#     def get_indexer(self):
#         raise Exception("Don't call me, call my subclasses")

#     def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
#         """
#         Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
#         :param sentence: words in the example to featurize
#         :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
#         At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
#         :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
#         a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
#         structure you prefer, since this does not interact with the framework code.
#         """
#         raise Exception("Don't call me, call my subclasses")


# class UnigramFeatureExtractor(FeatureExtractor):
#     """
#     Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
#     and any additional preprocessing you want to do.
#     """
#     def __init__(self, indexer: Indexer):
#         raise Exception("Must be implemented")


# class BigramFeatureExtractor(FeatureExtractor):
#     """
#     Bigram feature extractor analogous to the unigram one.
#     """
#     def __init__(self, indexer: Indexer):
#         raise Exception("Must be implemented")


# class BetterFeatureExtractor(FeatureExtractor):
#     """
#     Better feature extractor...try whatever you can think of!
#     """
#     def __init__(self, indexer: Indexer):
#         raise Exception("Must be implemented")


# class SentimentClassifier(object):
#     """
#     Sentiment classifier base type
#     """
#     def predict(self, sentence: List[str]) -> int:
#         """
#         :param sentence: words (List[str]) in the sentence to classify
#         :return: Either 0 for negative class or 1 for positive class
#         """
#         raise Exception("Don't call me, call my subclasses")


# class TrivialSentimentClassifier(SentimentClassifier):
#     """
#     Sentiment classifier that always predicts the positive class.
#     """
#     def predict(self, sentence: List[str]) -> int:
#         return 1


# class PerceptronClassifier(SentimentClassifier):
#     """
#     Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
#     superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
#     modify the constructor to pass these in.
#     """
#     def __init__(self):
#         raise Exception("Must be implemented")


# class LogisticRegressionClassifier(SentimentClassifier):
#     """
#     Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
#     superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
#     modify the constructor to pass these in.
#     """
#     def __init__(self):
#         raise Exception("Must be implemented")


# def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
#     """
#     Train a classifier with the perceptron.
#     :param train_exs: training set, List of SentimentExample objects
#     :param feat_extractor: feature extractor to use
#     :return: trained PerceptronClassifier model
#     """
#     raise Exception("Must be implemented")


# def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
#     """
#     Train a logistic regression model.
#     :param train_exs: training set, List of SentimentExample objects
#     :param feat_extractor: feature extractor to use
#     :return: trained LogisticRegressionClassifier model
#     """
#     raise Exception("Must be implemented")


# def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
#     """
#     Main entry point for your modifications. Trains and returns one of several models depending on the args
#     passed in from the main method. You may modify this function, but probably will not need to.
#     :param args: args bundle from sentiment_classifier.py
#     :param train_exs: training set, List of SentimentExample objects
#     :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
#     process, but you should *not* directly train on this data.
#     :return: trained SentimentClassifier model, of whichever type is specified
#     """
#     # Initialize feature extractor
#     if args.model == "TRIVIAL":
#         feat_extractor = None
#     elif args.feats == "UNIGRAM":
#         # Add additional preprocessing code here
#         feat_extractor = UnigramFeatureExtractor(Indexer())
#     elif args.feats == "BIGRAM":
#         # Add additional preprocessing code here
#         feat_extractor = BigramFeatureExtractor(Indexer())
#     elif args.feats == "BETTER":
#         # Add additional preprocessing code here
#         feat_extractor = BetterFeatureExtractor(Indexer())
#     else:
#         raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

#     # Train the model
#     if args.model == "TRIVIAL":
#         model = TrivialSentimentClassifier()
#     elif args.model == "PERCEPTRON":
#         model = train_perceptron(train_exs, feat_extractor)
#     elif args.model == "LR":
#         model = train_logistic_regression(train_exs, feat_extractor)
#     else:
#         raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
#     return model