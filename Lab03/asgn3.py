import numpy as np
from torchvision.datasets import MNIST
import time

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

#normalize the data
train_X = train_X / 255.0
test_X = test_X / 255.0

#convert the labels to one-hot-encoding.
train_Y_encoding = np.zeros((train_Y.size, 10))
train_Y_encoding[np.arange(train_Y.size), train_Y] = 1

test_Y_encoding = np.zeros((test_Y.size, 10))
test_Y_encoding[np.arange(test_Y.size), test_Y] = 1

#x 60 000* 784
#784 input neurons, 100 hidden neurons, and 10 output neurons
np.random.seed(40)
def initialization():
    W1 = np.random.randn(784, 100) * 0.01
    b1 = np.zeros((1, 100))
    W2 = np.random.randn(100, 10) * 0.01
    b2 = np.zeros((1, 10))
    return W1, b1, W2, b2
W1, b1, W2, b2 = initialization()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True)) #slide 30 curs 5
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2): #slide 35 curs 4 sau pseudocodul generalizat-48 curs 5
    z1 = np.dot(X, W1) + b1
    A1 = sigmoid(z1)
    z2 = np.dot(A1, W2) + b2
    A2 = softmax(z2)
    return z1, A1, z2, A2

def forward_dropout(X, W1, b1, W2, b2, dropout_rate=0.2): #80% raman activi
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    dropout_mask = (np.random.rand(*A1.shape) > dropout_rate).astype(float)  # masca dropout
    A1 *= dropout_mask  # dezactivam neuronii

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def cross_entropy(t_true, y_pred):
    return -np.mean(np.sum(t_true * np.log(y_pred), axis=1)) #slide 21 curs 5

def cross_entropy_regularization(t_true, y_pred, W1, W2, lambbda = 0.01):
    m = t_true.shape[0]
    cross_entropy_loss = cross_entropy(t_true, y_pred)
    l2_regularization = (lambbda / (2 * m)) * (np.sum(W1 ** 2) + np.sum(W2 ** 2))
    return cross_entropy_loss + l2_regularization

def backward(X, Y, A1, A2, W1, W2 , b1, b2, learning_rate=0.01, lambbda=0.01):
    m = X.shape[0]

    #slide 36,curs5 -dem pt formula
    dZ2 = A2 - Y  #eroarea pt final layer, formula slide 44, curs5 #derivata softmax
    dW2 = np.dot(A1.T, dZ2)  / m + (lambbda * W2 / m)  #formula slide 46
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m  #slide 47 zice ca e dZ2 ul

    # Gradient pentru stratul ascuns
    dA1 = np.dot(dZ2, W2.T)  #eroarea propagata, slide 45
    dZ1 = dA1 * A1 * (1 - A1)  #eroarea finala= eroarea propagata*deriv sigm;  deriv(sigm)=sigm*(1-sigm)(slide 87,curs4)
    # slide 6, curs 5
    dW1 = np.dot(X.T, dZ1) / m + (lambbda * W1 / m)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Actualizare parametri conform gradient descent
    #slide 83, curs4
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    return W1, b1, W2, b2

def accuracy(t_true, y_pred):
    pred_label = np.argmax(y_pred, axis=1)
    true_label = np.argmax(t_true, axis=1)
    return np.mean(pred_label == true_label)

def train(X, Y, W1, b1, W2, b2, epochs=200, batch_size=70, learning_rate=0.03):
    patience = 3
    decay_rate = 0.8
    best_accuracy = 0
    epochs_without_improvement = 0

    m= X.shape[0]
    epoch = 0
    start_time = time.time()
    max_time = 6*60
    while epoch < epochs:
        # shuffle data
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = Y[i:i+batch_size]

            Z1, A1, Z2, A2 = forward_dropout(X_batch, W1, b1, W2, b2)
            #loss = cross_entropy_regularization(y_batch, A2, W1, W2)
            W1, b1, W2, b2 = backward(X_batch, y_batch, A1, A2, W1, W2, b1, b2, learning_rate)

        _, _, _, A2_train = forward_dropout(X, W1, b1, W2, b2)
        _, _, _, A2_test = forward(test_X, W1, b1, W2, b2)

        train_loss = cross_entropy_regularization(Y, A2_train, W1, W2)
        train_acc = accuracy(Y, A2_train)
        test_acc = accuracy(test_Y_encoding, A2_test)

        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs} - Train Accuracy: {train_acc*100:.2f}% - Test Accuracy: {test_acc*100:.2f}% - Loss: {train_loss:.4f} - Elapsed Time: {elapsed_time:.2f}")
       # if train_acc >= 0.95 and test_acc >= 0.95:
        #    print("Stop the accuracy is over 95")
        #    break
        if elapsed_time >= max_time:
            print("Stop due to time limit")
            break

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            learning_rate = learning_rate * decay_rate
            epochs_without_improvement = 0
            print(f"Reducing learning rate to {learning_rate}")
        epoch +=1

    return W1, b1, W2, b2

epochs = 700

W1_trained, b1_trained, W2_trained, b2_trained = train(train_X, train_Y_encoding, W1, b1, W2, b2, epochs=epochs)




