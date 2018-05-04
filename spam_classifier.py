import numpy as np 
from regression import *
from naive_bayes import *
from svm import *
from qda import *
from lda import *
from k_nearest_neighbor import *
# Change these variables to indicate which methods to use 
baseline = False
linear_rg = False
naive_bay = False
nn = False 
svm = False
QDA = False
LDA = False
logistic_reg = False
decision_tr = False 
kNN = True

def main():
    # X is an n x d matrix, where n is the number of samples and d is the number of features.
    # Each feature corresponds to an item in the bag of words weighted by TF-IDF.

    with np.load("email_vector_training.npz") as trainingData:
        X_training = trainingData["arr_0"]
        y_training = trainingData["arr_1"]
    
    with np.load("email_vector_test.npz") as testData:
        X_test = testData["arr_0"]
        y_test = testData["arr_1"]
            
    PCA_flag = True
    CCA_flag = False

    if PCA_flag and CCA_flag:
        raise ValueErrror("Please only use one of the two dimensionality reduction methods")

    #Should probably set PCA and CCA inside each block.
    if baseline:
        print(">>>>>>>>>>>" + "Baseline" + ">>>>>>>>>>>")
        pred_train = np.ones((X_training.shape[0], 1))
        pred_test = np.ones((X_test.shape[0], 1))
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if QDA:
        print(">>>>>>>>>>>" + "QDA" + ">>>>>>>>>>>")
        #start_time = time.time()
        qda = QDA_Classifier()
        qda.train(X_training, y_training)
        pred_train = qda.predict(X_training).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = qda.predict(X_test).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if LDA:
        print(">>>>>>>>>>>" + "LDA" + ">>>>>>>>>>>")
        lda = LDA_Classifier()
        lda.train(X_training, y_training)
        pred_train = lda.predict(X_training).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = lda.predict(X_test).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if linear_rg:
        print(">>>>>>>>>>>" + "Linear Regression" + ">>>>>>>>>>>")
        w = linear_regression(X_training, y_training)
        print("Training error is " + str(np.linalg.norm(y_training - (X_training@w + 1)/2)**2/X_training.shape[0]))
        pred_test = X_test @ w
        pred_test = (pred_test > 0).astype(int)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if naive_bay:
        print(">>>>>>>>>>>" + "Naive Bayes" + ">>>>>>>>>>>")
        if PCA_flag or CCA_flag:
            print("Naive Bayes does not supposet dimensionality reduction. Continuing onto the next method.")
        else:
            nb = Naive_Bayes_Classifier()
            nb.train(X_training, y_training)
            pred_train = nb.predict(X_training)
            print(pred_train)
            print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
            pred_test = nb.predict(X_test)
            print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
            print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if nn:
        pass
    if svm:
        print(">>>>>>>>>>>" + "SVM" + ">>>>>>>>>>>")
        svm_classifier = SVM_Classifier(0.1)
        svm_classifier.train(X_training, y_training)
        pred_train = (svm_classifier.predict(X_training) > 0).astype(int).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = (svm_classifier.predict(X_test) > 0).astype(int).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if logistic_reg:
        print(">>>>>>>>>>>" + "Logistic Regression" + ">>>>>>>>>>>")
        lr = logistic_regression(X_training, y_training)
        pred_train = lr.predict(X_training).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = lr.predict(X_test).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if decision_tr:
        pass
    if kNN:
        print(">>>>>>>>>>>" + "K Nearest Neighbors" + ">>>>>>>>>>>")
        knn_classifier = KNearestNeighbor()
        knn_classifier.train(X_training, y_training)
        pred_train = knn_classifier.predict(X_training, 5)
        print(pred_train)
 
if __name__ == '__main__':
    main()