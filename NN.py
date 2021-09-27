from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_digits
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

    parameters = {'solver':('lbfgs', 'sgd', 'adam'), 'alpha':[0.0001, 0.001, 0.01]}
    nn = MLPClassifier(random_state=1, max_iter=500)

    clf1 = GridSearchCV(nn, parameters)
    clf1 = clf1.fit(df1_x, df1_y)
    print("Digits Best Params nn:")
    print(clf1.best_params_)

    clf2 = GridSearchCV(nn, parameters)
    clf2 = clf2.fit(df2_x, df2_y)
    print("LoL Best Params nn:")
    print(clf2.best_params_)
    print(clf2.best_score_)

    train_sizes, train_scores, valid_scores1 = learning_curve(MLPClassifier(random_state=1, 
                                                                                        solver="lbfgs",
                                                                                        max_iter=300, 
                                                                                        alpha=0.01),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(MLPClassifier(random_state=1, 
                                                                                        solver="lbfgs",
                                                                                        max_iter=300, 
                                                                                        alpha=0.01),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df2_x, y=df2_y, cv=5)

    plt.plot(np.linspace(0.1, 1, 10), np.mean(valid_scores1, axis=1), 'b')
    plt.fill_between(np.linspace(0.1, 1, 10), np.mean(valid_scores1, axis=1) - np.std(np.mean(valid_scores1, axis=1)),
                         np.mean(valid_scores1, axis=1) + np.std(np.mean(valid_scores1, axis=1)), alpha=0.1,
                         color="b")
    plt.plot(np.linspace(0.1, 1, 10), np.mean(valid_scores2, axis=1), 'r')
    plt.fill_between(np.linspace(0.1, 1, 10), np.mean(valid_scores2, axis=1) - np.std(np.mean(valid_scores2, axis=1)),
                         np.mean(valid_scores2, axis=1) + np.std(np.mean(valid_scores2, axis=1)), alpha=0.1,
                         color="r")
    plt.grid(linestyle='--')
    plt.title("Neural Network Learning Curve - Validation Scores (Digits Hyperparameters)")
    plt.xlabel("Training Examples [Percentage of Dataset]")
    plt.ylabel("Classification Score")
    plt.legend(["Digits-dataset", "LoL-dataset"])
    plt.savefig("NN-1.png")

    train_sizes, train_scores, valid_scores1 = learning_curve(MLPClassifier(random_state=1, 
                                                                                        solver="lbfgs",
                                                                                        max_iter=300, 
                                                                                        alpha=0.0001),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(MLPClassifier(random_state=1, 
                                                                                        solver="lbfgs",
                                                                                        max_iter=300, 
                                                                                        alpha=0.0001),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df2_x, y=df2_y, cv=5)

    plt.clf()
    plt.plot(np.linspace(0.1, 1, 10), np.mean(valid_scores1, axis=1), 'b')
    plt.fill_between(np.linspace(0.1, 1, 10), np.mean(valid_scores1, axis=1) - np.std(np.mean(valid_scores1, axis=1)),
                         np.mean(valid_scores1, axis=1) + np.std(np.mean(valid_scores1, axis=1)), alpha=0.1,
                         color="b")
    plt.plot(np.linspace(0.1, 1, 10), np.mean(valid_scores2, axis=1), 'r')
    plt.fill_between(np.linspace(0.1, 1, 10), np.mean(valid_scores2, axis=1) - np.std(np.mean(valid_scores2, axis=1)),
                         np.mean(valid_scores2, axis=1) + np.std(np.mean(valid_scores2, axis=1)), alpha=0.1,
                         color="r")
    plt.grid(linestyle='--')
    plt.title("Neural Network Learning Curve - Validation Scores (LoL Hyperparameters)")
    plt.xlabel("Training Examples [Percentage of Dataset]")
    plt.ylabel("Classification Score")
    plt.legend(["Digits-dataset", "LoL-dataset"])
    plt.savefig("NN-2.png")

    # train_sizes, train_scores, valid_scores1 = learning_curve(MLPClassifier(random_state=1, 
    #                                                                                     solver="lbfgs",
    #                                                                                     alpha=0.0001,
    #                                                                                     activation='identity'),
    #                                                                                     train_sizes=np.linspace(0.1, 1, 10),
    #                                                                                     random_state=1, 
    #                                                                                     X=df2_x, y=df2_y, cv=2)
    # train_sizes, train_scores, valid_scores2 = learning_curve(MLPClassifier(random_state=1, 
    #                                                                                     solver="lbfgs",
    #                                                                                     alpha=0.0001,
    #                                                                                     activation='logistic'),
    #                                                                                     train_sizes=np.linspace(0.1, 1, 10),
    #                                                                                     random_state=1, 
    #                                                                                     X=df2_x, y=df2_y, cv=2)
    # train_sizes, train_scores, valid_scores3 = learning_curve(MLPClassifier(random_state=1, 
    #                                                                                     solver="lbfgs",
    #                                                                                     alpha=0.0001,
    #                                                                                     activation='tanh'),
    #                                                                                     train_sizes=np.linspace(0.1, 1, 10),
    #                                                                                     random_state=1, 
    #                                                                                     X=df2_x, y=df2_y, cv=2)
    # train_sizes, train_scores, valid_scores4 = learning_curve(MLPClassifier(random_state=1, 
    #                                                                                     solver="lbfgs",
    #                                                                                     alpha=0.0001,
    #                                                                                     activation='relu'),
    #                                                                                     train_sizes=np.linspace(0.1, 1, 10),
    #                                                                                     random_state=1, 
    #                                                                                     X=df2_x, y=df2_y, cv=2)
                                                                                                                                                                                                                                                          

    # plt.plot(np.linspace(0.1, 1, 10), valid_scores1[:, 0], 'b')
    # plt.plot(np.linspace(0.1, 1, 10), valid_scores2[:, 0], 'r')
    # plt.plot(np.linspace(0.1, 1, 10), valid_scores3[:, 0], 'g')
    # plt.plot(np.linspace(0.1, 1, 10), valid_scores4[:, 0], 'm')

    # plt.grid(linestyle='--')
    # plt.title("Digits-Dataset Learning Curve - Validation Scores (Activation Functions)")
    # plt.xlabel("Training Examples [Percentage of Dataset]")
    # plt.ylabel("Classification Score")
    # plt.legend(["identity", "logistic", "tanh", "relu"])
    # plt.savefig("NN-3.png")


if __name__ == "__main__":
    main()