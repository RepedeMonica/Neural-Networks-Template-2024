import numpy as np
from torchvision.datasets import MNIST

#1. Load the MNIST dataset
def download_mnist(is_train: bool):
    dataset = MNIST(
        root='./data',
        transform=lambda x: np.array(x).flatten(),
        download=True,
        train=is_train
    )
    mnist_data = []
    mnist_labels = []
    for image, label in dataset:
        mnist_data.append(image)
        mnist_labels.append(label)
    return np.array(mnist_data), np.array(mnist_labels)
train_X, train_Y = download_mnist(True)
test_X, test_Y = download_mnist(False)

# normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

#convert the labels to one-hot-encoding.
train_Y_encoding = np.zeros((train_Y.size, 10))
train_Y_encoding[np.arange(train_Y.size), train_Y] = 1

test_Y_encoding = np.zeros((test_Y.size, 10))
test_Y_encoding[np.arange(test_Y.size), test_Y] = 1

# initialize weights and biases
np.random.seed(50)
W = np.random.randn(784, 10) * 0.01  
b = np.zeros((10,))  

def softmax(z): #2
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward_propagation(X, W, b): #1
    z = np.dot(X, W) + b  
    return softmax(z)  

#  cross-entropy loss
def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))  

# gradient descent
def update_weight_bias(X, y_true, y_pred, W, b, learning_rate):
    error = y_true - y_pred  # Target - y
    # W = W + μ * (Target - y) * X^T
    W += learning_rate * np.dot(X.T, error)
    # b = b + μ * (Target - y)
    b += learning_rate * np.mean(error, axis=0)
    return W, b

# accuracy
def compute_accuracy(y_true, y_pred):
    pred_labels = np.argmax(y_pred, axis=1)  
    true_labels = np.argmax(y_true, axis=1) 
    return np.mean(pred_labels == true_labels)  

# train
def train_perceptron(train_X, train_Y_encoding, test_X, test_Y_encoding, W, b, epochs=100, batch_size=100, learning_rate=0.01):
    m = train_X.shape[0]  

    initial_train_pred = forward_propagation(train_X, W, b)
    initial_test_pred = forward_propagation(test_X, W, b)
    
    initial_train_accuracy = compute_accuracy(train_Y_encoding, initial_train_pred)
    initial_test_accuracy = compute_accuracy(test_Y_encoding, initial_test_pred)
    
    print(f'Initial train Accuracy: {initial_train_accuracy*100:.2f}% & Initial test Accuracy: {initial_test_accuracy*100:.2f}%')
    

    for epoch in range(epochs):
        #shuffle data
        indices = np.arange(m)
        np.random.shuffle(indices)
        train_X = train_X[indices]
        train_Y_encoding = train_Y_encoding[indices]

        epoch_loss = 0

        # split data in mini-batch
        for i in range(0, m, batch_size):
            X_batch = train_X[i:i+batch_size]
            y_batch = train_Y_encoding[i:i+batch_size]
            
            y_pred = forward_propagation(X_batch, W, b)

            batch_loss = cross_entropy_loss(y_batch, y_pred)
            epoch_loss += batch_loss  
            
            W, b = update_weight_bias(X_batch, y_batch, y_pred, W, b, learning_rate)
        
        # accuracy on each epoch
        average_epoch_loss = epoch_loss / (m / batch_size)

        train_pred = forward_propagation(train_X, W, b)
        test_pred = forward_propagation(test_X, W, b)
        
        train_accuracy = compute_accuracy(train_Y_encoding, train_pred)
        test_accuracy = compute_accuracy(test_Y_encoding, test_pred)
        
        print(f'Epoch {epoch+1}/{epochs} => Train Accuracy: {train_accuracy*100:.2f}% & Test Accuracy: {test_accuracy*100:.2f}% & Average loss: {average_epoch_loss:.4f}')
    
    return W, b

epochs = 55  
batch_size = 100
learning_rate = 0.01

W_trained, b_trained = train_perceptron(train_X, train_Y_encoding, test_X, test_Y_encoding, W, b, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

test_pred_final = forward_propagation(test_X, W_trained, b_trained)
final_test_accuracy = compute_accuracy(test_Y_encoding, test_pred_final)
print(f'Final test Accuracy: {final_test_accuracy*100:.2f}%')
