import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def generate_examples(nb = 200):
    x = np.random.uniform(low = -10, high = 10, size=nb)
    y = 2 * x + np.ones(nb) * 3
    x += np.random.normal(0, 0.5, size=x.shape)
    y += np.random.normal(0, 0.5, size=y.shape)
    return x, y

def plot(x, y):
    plt.plot(x, y, '.')
    plt.show()

def plot_apprx(x, y, x_l, y_l):
    plt.plot(x, y, '.')
    plt.plot(x_l, y_l)
    plt.show()

def plt_history(history):
    plt.plot([L[0] for L in history], label='train loss')
    plt.plot([L[1] for L in history], label='val loss')
    plt.show()

def mse(y_hat, y):
    return np.sum(np.square(y_hat - y))/y.shape[0]

if __name__ == '__main__':
    x, y = generate_examples(1000)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
    #plot(x, y)
    W = np.random.rand(1)
    b = np.random.rand(1)
    lr = 0.001
    history = []
    for i in range(10000):
        predict = W*x_train + b
        dW = np.mean((predict - y_train) * x_train)
        db = np.mean(predict - y_train)
        W = W - lr * dW
        b = b - lr * db
        Loss = mse(predict, y_train)
        vLoss = mse(W*x_test + b, y_test)
        history.append((Loss, vLoss))
    print(W, b)
    x_l = [-10, 0, 10]
    y_l = x_l * W + b
    plot_apprx(x_train, y_train, x_l, y_l)
    plt_history(history)
