import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def load_data():
    data = np.loadtxt('danet.txt')
    y = data[:, 0]
    X = data[:, 1:]
    return data, X, y

def split_data(data, X, y):
    split_idx = int(len(data) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]

    print(f"Total samples: {len(data)}")
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")

    return X_train, X_test, y_train, y_test

def ar_model(X_train, y_train, X_test, y_test):

    v = np.linalg.pinv(X_train) @ y_train

    y_pred_train = X_train @ v
    y_pred_test = X_test @ v
    y_costless = X_test[:, 0]

    return v, y_pred_test, y_costless

def main():
    data, X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(data, X, y)

    v, y_pred_test, y_costless = ar_model(X_train, y_train, X_test, y_test)
    print("Test set verification:")
    print(f"{'Actual':<10} | {'AR Prediction':<15} | {'Costless':<10}")
    for i in range(len(y_test)):
        print(f"{y_test[i]:<10.4f} | {y_pred_test[i]:<15.4f} | {y_costless[i]:<10.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(y_test, 'r-o', label='Actual Quotes')
    plt.plot(y_pred_test, 'g--', label='AR Prediction')
    plt.plot(y_costless, 'b:', label='Costless Model (Lag 1)')
    plt.title("AR Model Verification (Test Set)")
    plt.legend()
    plt.show()

    print(data)


if __name__ == "__main__":
    main()