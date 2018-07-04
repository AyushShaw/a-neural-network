import numpy as np


def dataset():
    # Train data set
    X_train = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    lable = np.array([[0], [1], [1], [0]])
    return X_train, lable


def sigmoid(x):
    return (1/(1+np.exp(x)))


def sigderv(x):
    return x * (1 - x)


def main():
    np.random.seed(1)

    X, Y = dataset()
    epoch = 7000

    # Initializing weights
    W1 = 2*np.random.random((3, 4)) - 1
    W2 = 2*np.random.random((4, 1)) - 1

    for n in range(epoch):

        # Feedforward through layers layer1 being input layer
        layer1 = X
        layer2 = sigmoid(np.dot(layer1, W1))
        layer3 = sigmoid(np.dot(layer2, W2))

        # Finding Error
        layer3_error = Y - layer3
        # Noting Error @ every 100 epoch
        if (n % 100) == 0:
            print("Error: {}".format(np.mean(np.abs(layer3_error))))

        # Backprop the Error
        layer3_delta = layer3_error * sigderv(layer3)
        layer2_error = layer3_delta.dot(W2.T)
        layer2_delta = layer2_error * sigderv(layer2)

        # Updating Weights
        W2 += layer2.T.dot(layer3_delta)
        W1 += layer1.T.dot(layer2_delta)


if __name__ == '__main__':
    main()
