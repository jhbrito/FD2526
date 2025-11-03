import pandas as pd
from micromlgen import port
from sklearn.svm import SVC
import sys
from sklearn.preprocessing import LabelEncoder

from sklearn.datasets import load_iris


def load_data_from_csv(filename: str, label_column: str) -> tuple:
    """
    Convert csv file to X and y
    :param label_column:
    :param filename:
    :return:
    """
    df = pd.read_csv(filename)
    x_columns = [c for c in df.columns if c != label_column]
    X = df[x_columns].to_numpy(dtype=float)
    y_string = df[label_column]
    label_encoder = LabelEncoder().fit(y_string)
    y_numeric = label_encoder.transform(y_string)
    print('Label mapping', {label: i for i, label in enumerate(label_encoder.classes_)})

    return X, y_numeric


X, y = load_data_from_csv('iris.csv', label_column='variety')

clf = SVC(kernel='linear').fit(X, y)
clf.gamma = 0.001
svm_port = port(clf, classmap={0: 'setosa', 1: 'versicolor', 2: 'virginica'})

sys.stdout = open('IrisClassifier.h', 'wt')
print(svm_port)
