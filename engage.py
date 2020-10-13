import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


def importdata():

    balance_data = pd.read_csv(r'Posts_Engagement_csv.csv',
                               sep=',', header=0)

    print("Dataset Length: ", len(balance_data))
    print("Dataset Shape: ", balance_data.shape)

    print("Dataset: ")
    print(balance_data)
    return balance_data


def splitdataset(balance_data):

    X = balance_data.values[:, 0:3]
    Y = balance_data.values[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=100)

    return X, Y, X_train, X_test, y_train, y_test


def train_entropy(X_train, X_test, y_train):

    clf_entropy = DecisionTreeClassifier(
        criterion="entropy", random_state=100,
        max_depth=4, min_samples_leaf=10)

    clf_entropy.fit(X_train, y_train)         
    return clf_entropy


def prediction(X_test, clf_object):

    y_pred = clf_object.predict(X_test)
    print(y_pred)
    return y_pred


def cal_accuracy(y_test, y_pred):

    print("Accuracy : ",
          accuracy_score(y_test, y_pred)*100)

    print("Report : ",
          classification_report(y_test, y_pred))


def main():

    data = importdata()
    X, Y, X_train, X_test, y_train, y_test = splitdataset(data)
    clf_entropy = train_entropy(X_train, X_test, y_train)

    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)

    followers = int(input('How many followers you have?\n'))
    hour = int(input('What time is it?\n'))
    interactions = int(input('Expectd interactions:\n'))

    if(interactions > 10*followers ):
        print("That might not happen")
    else:
        y_pred_query = prediction([[followers, hour, interactions]], clf_entropy)


if __name__ == "__main__":
    main()
