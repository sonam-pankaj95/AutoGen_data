# filename: train_logistic_regression.py

from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
iris = datasets.load_iris()

# Split the dataset into features (X) and target (y)
X = iris.data
y = iris.target

# Train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Predict the species for the given data point
new_data = [[5.1, 3.5, 1.4, 0.2]]
species_predicted = model.predict(new_data)

print(iris.target_names[species_predicted])