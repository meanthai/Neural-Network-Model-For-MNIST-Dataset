# Neural-Network-Model-For-MNIST-Dataset
This built-from-scratch neural network model, developed by Hoang Minh Thai, serves as an AI model for classifying 10 distinct digits (ranging from 0 to 9) of the MNIST Dataset.

# MNIST Dataset
The 'mnist_784' dataset contains 70,000 examples, consisting of both training and testing data. Each example is a 28x28-pixel image paired with a label.. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

# Pre-processing Methods
Furthermore, PCA method is also applied to pre-process the MNIST dataset by reducing the dimensionality of the given dataset in order to reduce noise and improve performance of the model.

# Training Result (compared to Scikit-learn model MLPClassifier)
The result after training: My own built-from-scratch model achieves a nearly absolute accuracy with 99,29% for the training set (56,000 examples) and approximately 97,41% for the testing set (14,000 examples) that yields better performance than the built-in MLPClassifier model from Scikit-learn library.

Moreover, my own model's training time, with the same number of training epochs and batch size, is significantly faster than the Scikit-learn model's training time (around ~3 times faster).  

![image](https://github.com/meanthai/Neural-Network-Model-For-MNIST-Dataset/assets/147926426/e3ccc86e-dc1d-4217-8dbc-4a8986086ba9)


# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

# License
This project is licensed under the MIT License.

# Acknowledgements
This project was inspired by Scikit-learn model MLPClassifier.

