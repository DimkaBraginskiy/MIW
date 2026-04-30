import numpy as np
import keras
from keras import layers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical



def main():
    # Model / data parameters
    input_shape = (32, 32, 3)

    # Load the data and split it between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

    # combine original data
    X = np.concatenate([X_train, X_test])
    y = np.concatenate([y_train,y_test])

    animals = [2, 3, 4, 5, 6, 7]
    vehicles = [0, 1, 8, 9]

    y = np.array([0 if label in animals else 1 for label in y.flatten()])
    y = to_categorical(y, 2)



    X = X.astype("float32") / 255



    # split 30% train, 70% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.7, random_state=42
    )

    model1 = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(2
                     , activation="softmax")
    ])

    model1.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model2 = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(2, activation="softmax")
    ])

    model2.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])

    model3 = (keras.Sequential(
        [
            keras.Input(shape=input_shape),

            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(2, activation="softmax"),
        ]
    ))

    model3.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


    #TRAIN
    histories = {}

    for name, model in zip(
            ["model1", "model2", "model3"],
            [model1, model2, model3]
    ):
        print(f"\nTraining {name}...\n")

        history = model.fit(
            X_train, y_train,
            epochs=10,
            validation_data=(X_test, y_test),
            verbose=1
        )

        histories[name] = history


    #COMPARE

    results = {}

    for name, model in zip(
            ["model1", "model2", "model3"],
            [model1, model2, model3]
    ):
        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        results[name] = acc


    for name, acc in results.items():
        print(f"{name}: Test Accuracy = {acc:.4f}")


if __name__ == "__main__":
    main()
