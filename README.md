# Neural-Network-Model-For-MNIST-Dataset
A built-from-scratch Neural Network model by Hoang Minh Thai for classifying 10 different digits (a number from 0 to 9) from the examples of the MNIST Dataset

The given MNIST Dataset "mnist_test.csv" contains 10,000 test examples which are 28x28-pixels images and labels. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

Furthermore, PCA is also applied to pre-process the dataset by reducing the dimensionality of the given dataset in order to reduce noise and improve performance.

The result after training: The model achieving an absolute accuracy with 100% for the training set (8,000 examples) and approximately 95,5% for the testing set (2,000 examples). My own built-from-scratch model yields better performance than the MLPClassifier model from scikitlearn library. 

![image](https://github.com/meanthai/Neural-Network-Model-For-MNIST-Dataset/assets/147926426/a2050525-50f8-40e4-8244-08149bc68a91)
