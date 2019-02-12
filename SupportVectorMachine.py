from Classifier import Classifier
from sklearn.svm import SVC


# This is a subclass that extends the abstract class Classifier.
class SupportVectorMachine(Classifier):

    # The abstract method from the base class is implemeted here to return SVM classifier
    def buildClassifier(self, X_features, Y_train):
        print "Using SVM classifier..."
        clf = SVC().fit(X_features, Y_train)
        return clf
