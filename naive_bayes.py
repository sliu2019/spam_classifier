import numpy as np 
class Naive_Bayes_Classifier:
    def train(self, X, y):
        X = (X > 0).astype(int)
        X_y_0 = X[np.where(y == 0)[0], :]
        X_y_1 = X[np.where(y > 0)[0], :]
        self.probabilities_0 = (np.sum(X_y_0, axis = 0)/X_y_0.shape[0]).reshape(-1)
        self.probabilities_1 = (np.sum(X_y_1, axis = 0)/X_y_1.shape[0]).reshape(-1)
        self.y_prob = np.sum(y, axis = 0)/y.shape[0]
        
    def predict(self, X):
        X = (X > 0).astype(int)
        X_1_s = X
        X_0_s = np.absolute(X - 1)
        class_1_prob = np.multiply(X_1_s, self.probabilities_1) + np.multiply(X_0_s, (1 - self.probabilities_1))
        class_0_prob = np.multiply(X_1_s, self.probabilities_0) + np.multiply(X_0_s, (1 - self.probabilities_0))
        return ((self.y_prob * np.prod(class_1_prob, axis = 1)) > ((1 - self.y_prob) * np.prod(class_0_prob, axis = 1))).astype(int).reshape(-1, 1)

