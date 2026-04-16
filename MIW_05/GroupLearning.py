from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


def decision_tree(X_train, X_test, Y_train, Y_test):

    for criterion in ['gini', 'entropy']:
        for depth in [2, 5, 10, None]:
            clf = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=depth,
                random_state=42
            )
            clf.fit(X_train, Y_train)

            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)

            train_accuracy = accuracy_score(Y_train, y_train_pred)
            test_accuracy = accuracy_score(Y_test, y_test_pred)

            print(f"{criterion=} {depth=}")
            print(f"Train: {train_accuracy:.4f} Test: {test_accuracy:.4f}\n")

def random_forest(X_train, X_test, y_train, y_test):
    for n in [10,50,100]:

        clf = RandomForestClassifier(
            n_estimators=n,
            random_state=42
        )

        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        print(f"Trees={n}")
        print(f"Train: {train_accuracy:.4f} Test: {test_accuracy:.4f}\n")

def logistic_regression(X_train, X_test, Y_train, Y_test):

    #0.01 - underfitting
    #1 - best one
    #10 - overfitting

    for C in [0.01, 0.1, 1, 10]:

        clf = LogisticRegression(
            C=C,
            max_iter=1000
        )

        clf.fit(X_train, Y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        train_accuracy = accuracy_score(Y_train, y_train_pred)
        test_accuracy = accuracy_score(Y_test, y_test_pred)


        print(f"{C=}")
        print(f"Train: {train_accuracy:.4f} Test: {test_accuracy:.4f}\n")


def support_vector_machine(X_train, X_test, Y_train, Y_test):

    for C in [0.01, 0.1, 1, 10]:

        model = SVC(
            C=C
        )

        model.fit(X_train, Y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_accuracy = accuracy_score(Y_train, y_train_pred)
        test_accuracy = accuracy_score(Y_test, y_test_pred)

        print(f"{C=}")
        print(f"Train: {train_accuracy:.4f} Test: {test_accuracy:.4f}\n")


def main():
    X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    print("Decision Tree Classifier:")
    decision_tree(X_train, X_test, Y_train, Y_test)
    print("-" * 30)

    print("Random Forest:")
    random_forest(X_train, X_test, Y_train, Y_test)
    print("-" * 30)

    print("Logistic Regression:")
    logistic_regression(X_train, X_test, Y_train, Y_test)
    print("-" * 30)

    print("Support Vector Machine(SVM):")
    support_vector_machine(X_train, X_test, Y_train, Y_test)
    print("-" * 30)




if __name__ == "__main__":
    main()