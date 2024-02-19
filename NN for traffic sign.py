import numpy as np
from model import NeuralNetwork

# PART 1 - LOAD AND PREPROCESS DATA

train_data = np.loadtxt("data/mnist_train.csv", delimiter=',')
test_data = np.loadtxt("data/mnist_test.csv", delimiter=',')

train_labels = train_data[:, 0]
train_images = train_data[:, 1:]

X_train = train_images[(train_labels == 2) | (train_labels == 7)] / 255.
X_train = X_train.T

y_train = train_labels[(train_labels == 2) | (train_labels == 7)]
y_train[y_train == 2] = 0
y_train[y_train == 7] = 1


test_labels = test_data[:, 0]
test_images = test_data[:, 1:]

X_test = test_images[(test_labels == 2) | (test_labels == 7)] / 255.
X_test = X_test.T

y_test = test_labels[(test_labels == 2) | (test_labels == 7)]
y_test[y_test == 2] = 0
y_test[y_test == 7] = 1


# PART 2 - TRAIN THE MODEL
model = NeuralNetwork(n_nodes=300, lr=0.001, input_dim=X_train.shape[0])
model.train(X_train, y_train, n_iter=300)

# PART 3 - EVALUATE THE MODEL
y_pred, y_proba = model.predict(X_test)
test_loss = model.compute_loss(y_proba, y_test)
test_accuracy = model.compute_accuracy(y_pred, y_test)

print("The loss on the test set is:", test_loss)
print("The test accuracy is:", test_accuracy)
