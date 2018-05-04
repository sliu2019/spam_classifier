import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class Boost_Classifier():

	trained = False

	def __init__(self, base_estimator_in = DecisionTreeClassifier(max_depth=2), 
						n_estimators_in = 600, 
						learning_rate_in = 1):
		
		self.boost_predictor = AdaBoostClassifier(base_estimator = base_estimator_in, 
												n_estimators = n_estimators_in,
												learning_rate = learning_rate_in)

	def train(self, X_train, y_train):
		self.scaler = StandardScaler()  
		self.scaler.fit(X_train)  
		X_train = self.scaler.transform(X_train)  

		self.boost_predictor.fit(X_train, y_train)

		self.trained = True

		# return train accuracy
		return self.boost_predictor.score(X_train, y_train)


	def predict(self, X_test):
		assert self.trained,"You should call train first."
		X_test = self.scaler.transform(X_test)
		return self.boost_predictor.predict(X_test)
