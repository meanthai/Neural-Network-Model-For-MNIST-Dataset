from sklearn import datasets
from sklearn.datasets import fetch_openml

# Slicing the dataset to seperate the labels array and the features array
mnist_images, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True)
X = np.array(mnist_images, dtype=int)
y = np.array(mnist_labels, dtype=int)
