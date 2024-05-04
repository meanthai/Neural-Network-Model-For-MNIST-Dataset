# Neural-Network-Model-For-MNIST-Dataset
This is a neural network model built from scratch by Hoang Minh Thai. It is designed to classify 10 different digits (numbers from 0 to 9) using examples from the MNIST dataset.

# MNIST Dataset
The 'mnist_784' dataset contains 70,000 examples, consisting of both training and testing data. Each example is a 28x28-pixel image paired with a label.. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

# Pre-processing Methods
Furthermore, PCA method is also applied to pre-process the MNIST dataset by reducing the dimensionality of the given dataset in order to reduce noise and improve performance.

# Training Result (compared with scikitlearn model MLPClassifier)
The result after training: My own built-from-scratch model achieves a nearly absolute accuracy with 99,29% for the training set (56,000 examples) and approximately 97,41% for the testing set (14,000 examples) that yields better performance than the built-in MLPClassifier model from scikitlearn library. Moreover, my own model's training time, with the same number of training epochs and batch size, is significantly faster than the scikitlearn model's training time (around ~3 times faster).  

![image](https://github.com/meanthai/Neural-Network-Model-For-MNIST-Dataset/assets/147926426/0e214f1f-af85-4e57-9928-1aadec31c9fd)

