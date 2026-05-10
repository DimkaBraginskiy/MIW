import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import plotka


class LogisticRegressionGD(object):
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # Training phase:
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.W_ = rgen.normal(loc=0.0, scale=0.01, size=(n_features, n_classes)) # weight matrix
        self.b_ = np.zeros(n_classes) #bias for each class
        net_vector = []

        for i in range(self.n_iter):
            net = np.dot(X, self.W_) + self.b_

            #output = self.activation(net_input1 + net_input2 + net_input3)
            probs = self.activation(net)

            #errors = (y - output)
            errors = np.zeros_like(probs)
            for id, label in enumerate(y):
                errors[id, label] = 1
            errors -= probs


            self.W_ += self.eta * X.T.dot(errors) / n_samples
            self.b_ += self.eta * errors.sum(axis=0) / n_samples
            #cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z): #probability
        z_stable = z - np.max(z, axis=1, keepdims=True)  # stability trick
        exp_z = np.exp(z_stable)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def predict_proba(self, X):
        net = np.dot(X, self.W_) + self.b_
        return self.activation(net)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

        # konfiguruje generator znaczników i mapę kolorów
        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        # rysuje wykres powierzchni decyzyjnej
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        # rysuje wykres wszystkich próbek
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl,
                        edgecolor='black')

def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    print('x_train')
    print(X_train)

    print('y_train')
    print(y_train)

    model = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
    model.fit(X_train, y_train)

    # Predicted probabilities
    probs = model.predict_proba(X_test)
    print("Probabilities for each class:\n", probs)

    # Predicted classes
    preds = model.predict(X_test)
    print("Predicted classes:\n", preds)
    print("True labels:\n", y_test)

    plt.figure(figsize=(8, 6))
    plot_decision_regions(X, y, classifier=model)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.legend(loc='upper left')
    plt.title('Softmax Logistic Regression Decision Regions')
    plt.show()




if __name__ == '__main__':
    main()
