import numpy as np
# Code inspired by Stanford CS231n's HW1 assignment

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.
    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1):
    """
    Predict labels for test data using this classifier.
    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.
    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    dists = self.compute_distances(X)
    return self.predict_labels(dists, k=k)

  def compute_distances(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.
    Input / Output: Same as compute_distances_two_loops
    """
    # Below works on the principle that (x-y)^2 = x^2 - 2xy + y^2, and array broadcasting
    n_X = X.shape[0]
    n_X_train = self.X_train.shape[0]
    x_2 = np.reshape(np.sum(X**2, axis=1), (n_X, 1))
    y_2 = np.reshape(np.sum(self.X_train**2, axis=1), (1, n_X_train))
    dists = x_2 + y_2- 2 * X @ self.X_train.T
    
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.
    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.
    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
      
      top_k_labels = self.y_train[np.argsort(dists[i, :])[:k]]
      y_pred[i] = np.argmax(np.bincount(np.reshape(top_k_labels, (top_k_labels.size))))


    return y_pred

def main():
  knn_classifier = KNearestNeighbor()

  X_train = np.random.rand(20, 30)*10
  y_train = np.random.randint(0, 2, (20, 1))
  X_test = np.random.rand(10, 30)*10

  # knn_classifier.train(X_train, y_train)
  # y_hat = knn_classifier.predict(X_test, k = 5)
  # print(y_hat)

if __name__ == "__main__":
  main()