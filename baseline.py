import numpy as np 
class Baseline_Classifier:
    def train(self, X_training, y_training):
        self.ham_prob = np.sum(y_training, axis = 0)/y_training.shape[0]


    def predict(self, X_test):
        if self.ham_prob > 0.5:
            return np.ones((X_test.shape[0], 1))
        else:
            return np.zeros((X_test.shape[0], 1))