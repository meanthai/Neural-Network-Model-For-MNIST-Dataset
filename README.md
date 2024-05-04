# Neural-Network-Model-For-MNIST-Dataset
A built-from-scratch Neural Network model by Hoang Minh Thai for classifying 10 different digits (a number from 0 to 9) from the examples of the MNIST Dataset.

The given MNIST Dataset "mnist_784" contains 70,000 test examples which are 28x28-pixels images and labels. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

Furthermore, PCA method is also applied to pre-process the MNIST dataset by reducing the dimensionality of the given dataset in order to reduce noise and improve performance.

The result after training: My own built-from-scratch model achieves a nearly absolute accuracy with 99,29% for the training set (56,000 examples) and approximately 97,41% for the testing set (14,000 examples) that yields better performance than the built-in MLPClassifier model from scikitlearn library. Moreover, my own model's training time, with the same number of training epochs and batch size, is significantly faster than the scikitlearn model's training time.  

![image](https://github.com/meanthai/Neural-Network-Model-For-MNIST-Dataset/assets/147926426/0e214f1f-af85-4e57-9928-1aadec31c9fd)

