import pandas as pd
from sklearn.preprocessing import LabelEncoder
from everywhereml.sklearn.ensemble import RandomForestClassifier


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
clf = RandomForestClassifier(n_estimators=5).fit(X, y)

import sys
sys.stdout = open('IrisClassifier.h', 'wt')
print(clf.to_arduino(instance_name='irisClassifier'))
