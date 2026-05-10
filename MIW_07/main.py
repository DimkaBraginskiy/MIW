import numpy as np
import os
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.ops.losses.losses_impl import mean_squared_error


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

def rnn_model(X_train, y_train, X_test, y_test):
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

    # LSTM needs [samples, time_steps, features]
    X_train_3D = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_3D = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(X_train_3D.shape[1], X_train_3D.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train_3D, y_train_scaled, epochs=100, batch_size=1, verbose=0)

    y_pred_scaled = model.predict(X_test_3D)
    return scaler_y.inverse_transform(y_pred_scaled)


def plot_results(y_test, y_ar_pred, y_rnn_pred, y_costless):
    plt.figure(figsize=(12, 6))

    plt.plot(y_test, 'r-o', label='Actual Quotes', linewidth=2)
    plt.plot(y_ar_pred, 'g--', label='Linear AR Model', alpha=0.8)
    plt.plot(y_rnn_pred, 'k-', label='RNN (LSTM) Model', linewidth=2)
    plt.plot(y_costless, 'b:', label='Costless Model (Reference)')

    plt.title("Comparison of Stock Price Prediction Models")
    plt.xlabel("Time Steps (Test Set)")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def model_comparator(y_test, y_ar_pred, y_rnn_pred, y_costless):
    # Root Mean Squared Error calculation considering each model...
    rmse_ar = np.sqrt(mean_squared_error(y_test, y_ar_pred))
    rmse_rnn = np.sqrt(mean_squared_error(y_test, y_rnn_pred))
    rmse_base = np.sqrt(mean_squared_error(y_test, y_costless))

    print("=== model comparison ===")
    print(f"Autoregression model: {rmse_ar:.4f}")
    print(f"RNN model: {rmse_rnn:.4f}")
    print(f"Base model: {rmse_base:.4f}")

    models = {"AR": rmse_ar, "RNN": rmse_rnn, "Baseline": rmse_base}
    winner = min(models, key=models.get)
    print(f"\nBest Performing Model: {winner}")


def main():
    data, X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(data, X, y)

    v, y_pred_ar, y_costless = ar_model(X_train, y_train, X_test, y_test)

    y_rnn_pred = rnn_model(X_train, y_train, X_test, y_test)

    plot_results(y_test, y_pred_ar, y_rnn_pred, y_costless)

    model_comparator(y_test, y_rnn_pred, y_costless)

if __name__ == "__main__":
    main()