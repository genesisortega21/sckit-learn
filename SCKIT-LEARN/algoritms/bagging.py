import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./SCKIT-LEARN/in/DataFinal.csv')

    # Verifica la estructura del DataFrame
    print(dt_heart.head())

    x = dt_heart.drop(['INCIDENCIA'], axis=1)
    y = dt_heart['INCIDENCIA']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)

    # Verifica la estructura de los datos de entrenamiento
    print(X_train.head())

    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_prediction = knn_class.predict(X_test)
    print('=' * 64)
    print('SCORE con KNN: ', accuracy_score(knn_prediction, y_test))

    estimators = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'LinearSVC': LinearSVC(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN': KNeighborsClassifier(),
        'DecisionTreeClf': DecisionTreeClassifier(),
        'RandomTreeForest': RandomForestClassifier(random_state=0)
    }

    for name, estimator in estimators.items():
        best_n_estimators = 0
        best_score = 0

        for n_estimators in range(1, 101):  # Prueba con valores de n_estimators de 1 a 100
            bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators).fit(X_train, y_train)
            bag_predict = bag_class.predict(X_test)
            score = accuracy_score(bag_predict, y_test)

            if score > best_score:
                best_score = score
                best_n_estimators = n_estimators

        print('=' * 64)
        print(f'BEST SCORE Bagging with {name} (n_estimators={best_n_estimators}): {best_score}')
