import numpy as np 
from regression import *
from naive_bayes import *
from svm import *
from qda import *
from lda import *
import time
# Change these variables to indicate which methods to use 
linear_rg = False
naive_bay = False
nn = False 
ls_svm = False 
svm = False
QDA = False
LDA = False
logistic_reg = False
decision_tr = False 
kNN = False 

def main():
	# X is an n x d matrix, where n is the number of samples and d is the number of features.
	# Each feature corresponds to an item in the bag of words weighted by TF-IDF.

	with np.load("email_vector_training.npz") as trainingData:
		X_training = trainingData["arr_0"]
		y_training = trainingData["arr_1"]
	
	with np.load("email_vector_test.npz") as testData:
		X_test = testData["arr_0"]
		y_test = testData["arr_1"]
			

	#Should probably set PCA and CCA inside each block.
	if QDA:
		print(">>>>>>>>>>>" + "QDA" + ">>>>>>>>>>>")

		#start_time = time.time()
		qda = QDA_Classifier()
		qda.train(X_training, y_training)
		#print("finished training in ", (time.time() - start_time)/60.0, " minutes" )
		#before_pred_time = time.time()
		#pred_train = qda.predict(X_training)
		#print("finished predicting in ", (time.time() - before_pred_time)/60.0, " minutes")
		print(pred_train)
	if LDA:
		print(">>>>>>>>>>>" + "LDA" + ">>>>>>>>>>>")
		lda = LDA_Classifier()
		lda.train(X_training, y_training)
		pred_train = lda.predict(X_training)
		print(pred_train)
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
		pass 
 
if __name__ == '__main__':
	main()