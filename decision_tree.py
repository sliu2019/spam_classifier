import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Tree_Classifier():

	trained = False

	def __init__(self):
		self.tree_predictor = DecisionTreeClassifier()

	def train(self, X_train, y_train):
		self.scaler = StandardScaler()  
		self.scaler.fit(X_train)  
		X_train = self.scaler.transform(X_train)  
		self.tree_predictor.fit(X_train, y_train)
		self.trianed = True
		# return train accuracy
		return self.tree_predictor.score(X_train, y_train)

	def predict(self, X_test):
		assert self.trianed,"You should call train first."
		X_test = self.scaler.transform(X_test)
		return self.tree_predictor.predict(X_test)

	def plot(self, X_train, y_train, X_test, y_test):
		
		depths = range(1,10)
		accuracy = []
		
		for i in depths:	
			classifier = DecisionTreeClassifier(max_depth = i)
			scaler = StandardScaler()  
			scaler.fit(X_train)  
			X_train = scaler.transform(X_train)  
			classifier.fit(X_train, y_train)
			X_test = scaler.transform(X_test)
			accuracy.append(classifier.score(X_test, y_test))

		plt.plot(depths, accuracy)
		plt.xlabel("max_depth")
		plt.ylabel("accuracy")
		plt.title("regularization comparison in decision_tree")
		plt.savefig("decision_tree_analysis.png")
		plt.show()
