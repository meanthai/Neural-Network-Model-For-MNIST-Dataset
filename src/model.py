# Xavier Weight Initialization
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

class MLP:
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
