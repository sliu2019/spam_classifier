import numpy as np 
from regression import *
from naive_bayes import *
from svm import *
from qda import *
from lda import *
from k_nearest_neighbor import *
from baseline import *
from util import * 
from neural_network import *
from boost import *
from decision_tree import *
import matplotlib.pyplot as plt
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
kNN = False
boost = False

def dimension_search(classifier, dr_method, X_training, y_training, X_test, y_test, min_k = 1, max_k = 600, interval = 5, save = False, name = None):
    # This function searches for the best number of dimensions k that yields the highest test accuracies.
    # Input: classifier - an instantiated classifier that has method train() and predict()
    #        dr_method - a string of either "PCA", "CCA", or "Random"
    #        X_training, y_training, X_test, y_test - numpy arrays n * k
    #        save - a boolean variable indicating whether we will save the plot
    #        name - an additional string argument used to indicate the name of the classifier
    # Output: k - best number of dimensions to reduce the feature matrix to
    if dr_method == "PCA":
        Projector = PCA_Projector
    elif dr_method == "CCA":
        Projector = CCA_Projector
    elif dr_method == "Random":
        Projector = Random_Projector
    else:
        raise ValueError("Please input a valie projector type")
    training_accuracies = []
    test_accuracies = []
    ks = []
    for k in range(min_k, max_k+1, interval):
        U = Projector(X_training, y_training, k)
        X_training_hat = X_training @ U
        X_test_hat = X_test @ U 
        classifier.train(X_training_hat, y_training)
        pred_train = classifier.predict(X_training_hat)
        pred_test = classifier.predict(X_test_hat)
        ks.append(k)
        training_accuracies.append(1 - np.sum(np.absolute(y_training - pred_train))/X_training.shape[0])
        test_accuracies.append(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0])
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(ks, training_accuracies)
    axs[1].plot(ks, test_accuracies)
    fig.suptitle("Training Accuracy & Test Accuracy vs. k")
    if save:
        if name:
            plt.savefig(name + "_" + dr_method + ".jpg")
    else:
        plt.show()
    return ks[np.argmax(np.array(test_accuracies))]


def main():
    # X is an n x d matrix, where n is the number of samples and d is the number of features.
    # Each feature corresponds to an item in the bag of words weighted by TF-IDF.

    with np.load("email_vector_training.npz") as trainingData:
        X_training = trainingData["arr_0"]
        y_training = trainingData["arr_1"]
    
    with np.load("email_vector_test.npz") as testData:
        X_test = testData["arr_0"]
        y_test = testData["arr_1"]

    # Change these variables to select which dimensionality reduction method to use, 
    # and the number of dimension k to reduce to.
    PCA_flag = False
    CCA_flag = False
    Random_flag = False
    k = 2 

    if PCA_flag:
        V_k = PCA_Projector(X_training, k)
        X_training = X_training @ V_k
        X_test = X_test @ V_k
    elif CCA_flag:
        U = CCA_Projector(X_training, y_training, k)
        X_training = X_training @ U
        X_test = X_test @ U
    elif Random_flag:
        U = Random_Projector(X_training, k)
        X_training = X_training @ U 
        X_test = X_test @ U 

    # regularization on data
    add_noise = False
    if add_noise:
        rng = np.random.RandomState(2)
        X_training += rng.uniform(size=X_training.shape)
        X_test += rng.uniform(size=X_test.shape)

    if baseline:
        print(">>>>>>>>>>>" + "Baseline" + ">>>>>>>>>>>")
        baseline_classifier = Baseline_Classifier()
        baseline_classifier.train(X_training, y_training)
        pred_train = baseline_classifier.predict(X_training)
        pred_test = baseline_classifier.predict(X_test)
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
        print(">>>>>>>>>>>" + "LS-SVM" + ">>>>>>>>>>>")
        ls_svm_classifier = LS_SVM()
        ls_svm_classifier.train(X_training, y_training)
        pred_train = ls_svm_classifier.predict(X_training)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = ls_svm_classifier.predict(X_test)
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
            print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
            pred_test = nb.predict(X_test)
            print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
            print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if nn:
        print(">>>>>>>>>>>" + "Neural Network" + ">>>>>>>>>>>")
        y_training = y_training.ravel()

        nn_classifier = NN_Classifier()
        nn_classifier.train(X_training, y_training)
        nn_classifier.plot(X_training, y_training, X_test, y_test)
        pred_train = (nn_classifier.predict(X_training) > 0).astype(int).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = (nn_classifier.predict(X_test) > 0).astype(int).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))

    if boost:
        print(">>>>>>>>>>>" + "Boost" + ">>>>>>>>>>>")
        y_training = y_training.ravel()

        boost_classifier = Boost_Classifier()
        boost_classifier.train(X_training, y_training)
        pred_train = boost_classifier.predict(X_training).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = boost_classifier.predict(X_test).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
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
        lr_classifier = logistic_regression_classifier()
        lr_classifier.train(X_training, y_training)
        pred_train = lr_classifier.predict(X_training)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = lr_classifier.predict(X_test)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if decision_tr:
        print(">>>>>>>>>>>" + "Neural Network" + ">>>>>>>>>>>")
        y_training = y_training.ravel()

        tree_classifier = Tree_Classifier()
        tree_classifier.train(X_training, y_training)
        tree_classifier.plot(X_training, y_training, X_test, y_test)
        pred_train = (tree_classifier.predict(X_training) > 0).astype(int).reshape(-1, 1)
        print("Training error is " + str(np.linalg.norm(y_training - pred_train)**2/X_training.shape[0]))
        pred_test = (tree_classifier.predict(X_test) > 0).astype(int).reshape(-1, 1)
        print("Test Error is " + str(np.linalg.norm(y_test - pred_test)**2/X_test.shape[0]))
        print("Test accuracy is " + str(1 - np.sum(np.absolute(y_test - pred_test))/X_test.shape[0]))
    if kNN:
        print(">>>>>>>>>>>" + "K Nearest Neighbors" + ">>>>>>>>>>>")
        knn_classifier = KNearestNeighbor()
        knn_classifier.train(X_training, y_training)
        pred_train = knn_classifier.predict(X_training, 5)
        print(pred_train)


def run_all_dimensionality_test():
    with np.load("email_vector_training.npz") as trainingData:
        X_training = trainingData["arr_0"]
        y_training = trainingData["arr_1"]
    
    with np.load("email_vector_test.npz") as testData:
        X_test = testData["arr_0"]
        y_test = testData["arr_1"]
    names = ["LS_SVM", "SVM", "QDA", "LDA", "logistic_reg"]
    for name in names:
        if name == "LS_SVM":
            classifier = LS_SVM()
        elif name == "SVM":
            classifier = SVM_Classifier(0.1)
        elif name == "QDA":
            classifier = QDA_Classifier()
        elif name == "LDA":
            classifier = LDA_Classifier()
        elif name == "logistic_reg":
            classifier = logistic_regression_classifier()
        print(name)
        for dr_method in ["PCA", "CCA", "Random"]:
            print(dr_method)
            print("Best k is:")
            print(dimension_search(classifier, dr_method, X_training, y_training, X_test, y_test, min_k = 1, max_k = 600, interval = 30, save = True, name = name))

if __name__ == '__main__':
    #run_all_dimensionality_test()
    main()