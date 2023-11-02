# filename: iris_plot.py
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# load the iris dataset
iris = load_iris()

# extract the petal length and petal width data
petal_length = iris.data[:, 2]
petal_width = iris.data[:, 3]

# create a scatter plot
plt.scatter(petal_length, petal_width)

# add labels and title
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Petal Length vs Petal Width')

# display the plot
plt.show()