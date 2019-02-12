from Classifier import Classifier
from sklearn.neural_network import MLPClassifier


# This is a subclass that extends the abstract class Classifier.
class MultiLayerPerceptron(Classifier):

    # The abstract method from the base class is implemeted here to return SVM classifier
    def buildClassifier(self, X_features, Y_train):
        print "Using MLP classifier..."
        clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1)
        clf = clf.fit(X_features, Y_train)
        return clf
