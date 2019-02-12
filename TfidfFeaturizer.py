from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer


# This is a subclass that extends the abstract class Featurizer.
class TfidfFeaturizer(Featurizer):

    # The abstract method from the base class is implemeted here to return tf-idf features
    def getFeatureRepresentation(self, X_train, X_val):
        print "Using TfidfFeaturizer..."
        vectorizer = TfidfVectorizer()
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_val_tfidf = vectorizer.transform(X_val)
        return X_train_tfidf, X_val_tfidf
