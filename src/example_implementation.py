import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_openml

# PCA step by step explaination by Fredtou
def PCA(X):
    num_components = 0
    
    #Step-1: Apply normalization method
    # Scaling data using Z-score normalization
    scaler = StandardScaler()
    X_meaned = scaler.fit_transform(X)
    
    #Step-2: Creating covariance matrix
    cov_mat = np.cov(X_meaned, rowvar = False)
     
    #Step-3: Calculating eigen values and eigen vectors
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
     
    #Step-4: Sorting the eigen vectors in descending order based on the eigen values
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    total = sum(eigen_values)
    var_exp = [( i /total ) * 100 for i in sorted_eigenvalue]
    cum_var_exp = np.cumsum(var_exp)
    for ite, percentage in enumerate(cum_var_exp):
        if percentage >= 95: # Take the features that make the variance percentage over 95%
            num_components = ite
            print(ite)
            break
    # print("percentage of cummulative variance per eigenvector in order: ", cum_var_exp)
         
    #Step-5: Extracting the final dataset after applying dimensionality reduction
    eigenvector_subset = sorted_eigenvectors[:, : num_components]
     
    #Step-6:Transforming the processed matrix
    X_reduced = np.dot(eigenvector_subset.transpose() , X_meaned.transpose() ).transpose()
     
    return X_reduced


def xavier_init(input_size, output_size):
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, size=(input_size, output_size))

# Applying one-hot Encoding to transform the categories into categorical binary vectors suitable for machine learning
def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i, label] = 1
    return encoded_labels  

# Decoding categorical binary vectors back to the original labels
def one_hot_decode(y):
    return np.argmax(y, axis = 1)

# Using RelU activation (non-linear function) for each neuron in layers
def relu(value):
    return np.maximum(0, value)


# Derivative of relU activation
def relu_derivative(value):
    return np.where(value > 0, 1, 0)


# Using softmax to calculate the probs of each label for multi-categories classification
def softmax(X):
    # Numerically stable softmax
    eps = 1e-15
    exp_shifted = np.exp(X - np.max(X, axis=1, keepdims=True))
    softmax_output = exp_shifted / (np.sum(exp_shifted, axis=1, keepdims=True) + eps)
    return softmax_output

class Fredtou_model:
    def __init__(self, input_size, hidden_sizes, output_size, batch, learning_rate, alpha, epochs, tol):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.batch = batch
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.epochs = epochs
        self.tol = tol
        
        # Initializing weights and biases using Xavier weight Initialization
        self.weights = []
        self.biases = []
        sizes = [input_size] + hidden_sizes + [output_size]
        self.sizes = sizes
        for i in range(len(sizes) - 1):
            self.weights.append(xavier_init(sizes[i], sizes[i + 1])) # Xavier weight initialization
            self.biases.append(np.zeros((1, sizes[i + 1])))
    
    def cross_entropy_loss(self, probs, y_batch):
        # Compute cross-entropy loss
        eps = 1e-15
        y_batch_decoded = one_hot_decode(y_batch)
        correct_log_probs = -np.log(probs[range(len(y_batch)), y_batch_decoded] + eps)
        data_loss = np.sum(correct_log_probs) / len(y_batch)
        return data_loss
            
    def fit(self, X, y):
        num_examples = X.shape[0]
        for epoch in range(self.epochs):
            # Shuffle the data
            permutation = np.random.permutation(num_examples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            batch_loss = []
            
            # Taking a batch of examples to train
            for i in range(0, num_examples, self.batch):
                X_batch = X_shuffled[i : i + self.batch]
                y_batch = y_shuffled[i : i + self.batch]
            
                # Forward pass
                layer = X_batch
                activations = []
                layers = []
                
                for i in range(len(self.weights)):
                    layer = layer.dot(self.weights[i]) + self.biases[i] 
                    layers.append(layer)
                    if i < len(self.weights) - 1:
                        layer = relu(layer)
                    else:
                        layer = softmax(layer) # Applying softmax function for the last layer
                    activations.append(layer)
                    
                # Compute loss with L2 regularization
                scores = layers[-1]
                probs = softmax(scores)
                data_loss = self.cross_entropy_loss(probs, y_batch)
                
                if epoch % 1 == 0:
                    batch_loss.append(data_loss)
                
                # Backpropagation
                d_weights = []
                d_biases = []
                
                # Calculating the derivative of weights and biases of the first layer backwards
                error = (activations[-1] - y_batch)
                errors = [error]
                d_weights.append(activations[-2].T.dot(error))
                d_biases.append(np.sum(error, axis = 0, keepdims=True))
                
                # Looping through other layers backwards to calculate the derivative of weights and biases
                for i in range(len(self.sizes) - 3, -1, -1):
                    error = errors[-1].dot(self.weights[i + 1].T) * relu_derivative(layers[i])
                    errors.append(error)
                    if i - 1 < 0:
                        d_weights.append(X_batch.T.dot(errors[-1]))
                        d_biases.append(np.sum(error, axis = 0, keepdims=True))
                    else:
                        d_weights.append(activations[i - 1].T.dot(errors[-1]))
                        d_biases.append(np.sum(error, axis = 0, keepdims=True))
                        
                d_weights.reverse()
                d_biases.reverse()
                
                # Add gradients of regularization term alpha
                for i in range(len(d_weights)):
                    d_weights[i] += self.alpha * self.weights[i]
                
                # Update weights and biases
                for i in range(len(self.weights)):
                    self.weights[i] -= self.learning_rate * d_weights[i]
                    self.biases[i] -= self.learning_rate * d_biases[i]
            
            if epoch % 1 == 0:
                print(f"After {epoch}th epoch, the loss value is: ", np.mean(batch_loss, axis=0))
            if np.mean(batch_loss, axis=0) < self.tol:
                return
    
    # Multi-categories Classification
    def predict(self, X):
        layer = X
        for i in range(len(self.weights)):
            layer = layer.dot(self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                layer = relu(layer)

        probs = softmax(layer) # Applying softmax function to have the probs of every class
        return np.argmax(probs, axis=1)# Return the class with the highest probability for each sample


def show_image(imgs, labels):
    imgs = imgs.reshape((-1, 28, 28)) # Reshaping the images as they need to be two dimensional images
    plt.figure(figsize=(10, 5))
    
    num_imgs = 10
    
    # Plot the first (num_imgs) images of the dataset with their labels
    for i in range(num_imgs):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imgs[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

mnist_images, mnist_labels = fetch_openml('mnist_784', version=1, return_X_y=True)

# Slicing the dataset to seperate the labels array and the features array
X = np.array(mnist_images, dtype=int)
y = np.array(mnist_labels, dtype=int)

show_image(X, y) # Plotting images

# Applying PCA to scale the dataset and reduce dimensionality
X = PCA(X)

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Onehot encoding the labels
num_classes = len(set(y))
y_train_encoded = one_hot_encode(y_train, num_classes)
y_test_encoded = one_hot_encode(y_test, num_classes)

# Define and train the model
batch_size = 4
mlp = Fredtou_model(input_size=X.shape[1], output_size=num_classes, hidden_sizes = [284, 128, 16], alpha= 0.01, learning_rate=0.001, epochs=15, batch = batch_size, tol = 0.000001)
mlp.fit(X_train, y_train_encoded)

# Prediction and evaluation
y_predict_train = mlp.predict(X_train)
y_predict_test = mlp.predict(X_test)
train_accuracy = np.mean(y_predict_train == y_train)
test_accuracy = np.mean(y_predict_test == y_test)

print("Training accuracy of my own model:", train_accuracy)
print("Testing accuracy of my own model", test_accuracy)
print(classification_report(y_predict_test, y_test))


# Using MLPClassifier from scikitlearn library
model = MLPClassifier(solver="adam", alpha=0.01, learning_rate_init=0.001, hidden_layer_sizes=(284, 128, 16), batch_size=4 , max_iter=15, verbose=False, tol= 0.0000001)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_accuracy = np.mean(y_pred_train == y_train)
test_accuracy = np.mean(y_pred_test == y_test)

print("Training accuracy of the scikit learn model:", train_accuracy)
print("Testing accuracy of the scikit learn model:", test_accuracy)


