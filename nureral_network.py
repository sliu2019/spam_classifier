import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  


X, y = make_classification(n_features=20, n_redundant=3, n_informative=2,
							random_state=1, n_clusters_per_class=1)


def train(X,y)
	# replaced by our dataset
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	scaler = StandardScaler()  
	scaler.fit(X_train)  
	X_train = scaler.transform(X_train)  
	X_test = scaler.transform(X_test)

	mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
				solver='sgd', tol=1e-4, random_state=1)

	mlp.fit(X_train, y_train)
	train_err = mlp.score(X_train, y_train)
	test_err = mlp.score(X_test, y_test)
	print("Training set score: %f" % train_err)
	print("Test set score: %f" % test_err)

	return (train_err, test_err)




