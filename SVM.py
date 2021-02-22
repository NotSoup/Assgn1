from sklearn.datasets import load_digits
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
import numpy as np


def main():

    df1 = load_digits()
    df1_x = df1.data        # (8,8) image to (64,) array flattening
    df1_y = df1.target

    df2 = pd.read_csv("ds2.csv")
    df2_x = df2.drop(columns=["blue_win"])
    df2_y = df2["blue_win"]

    parameters = {'gamma':[0.0001, 0.001, 0.01], 'degree':[3, 4, 5]}
    svc = svm.SVC(random_state=1)

    clf1 = GridSearchCV(svc, parameters)
    clf1 = clf1.fit(df1_x, df1_y)
    print("Digits Best Params dt:")
    print(clf1.best_params_)

    clf2 = GridSearchCV(svc, parameters)
    clf2 = clf2.fit(df2_x, df2_y)
    print("LoL Best Params dt:")
    print(clf2.best_params_)

    train_sizes, train_scores, valid_scores1 = learning_curve(svm.SVC(random_state=1, 
                                                                                        gamma=0.001, 
                                                                                        degree=3),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(svm.SVC(random_state=1, 
                                                                                        gamma=0.001, 
                                                                                        degree=3),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df2_x, y=df2_y, cv=5)

    plt.plot(np.linspace(0.1, 1, 10), valid_scores1[:, 0])
    plt.plot(np.linspace(0.1, 1, 10), valid_scores2[:, 0])
    plt.title("SVC Learning Curve (Digits Hyperparameters)")
    plt.xlabel("Training Set Size [Percentage]")
    plt.ylabel("Test Set Score")
    plt.legend(["Digits", "LoL"])
    plt.savefig("SVC-1.png")

    train_sizes, train_scores, valid_scores1 = learning_curve(svm.SVC(random_state=1, 
                                                                                        gamma=0.0001, 
                                                                                        degree=3),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(svm.SVC(random_state=1, 
                                                                                        gamma=0.0001, 
                                                                                        degree=3),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df2_x, y=df2_y, cv=5)

    plt.clf()
    plt.plot(np.linspace(0.1, 1, 10), valid_scores1[:, 0])
    plt.plot(np.linspace(0.1, 1, 10), valid_scores2[:, 0])
    plt.title("SVC Learning Curve (LoL Hyperparameters)")
    plt.xlabel("Training Set Size [Percentage]")
    plt.ylabel("Test Set Score")
    plt.legend(["Digits", "LoL"])
    plt.savefig("SVC-2.png")

if __name__ == "__main__":
    main()