import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.model_selection import learning_curve



def main():

    df1 = load_digits()
    df1_x = df1.data        # (8,8) image to (64,) array flattening
    df1_y = df1.target

    df2 = pd.read_csv("ds2.csv")
    df2_x = df2.drop(columns=["blue_win"])
    df2_y = df2["blue_win"]

    parameters = {'learning_rate':[0.1, 0.3, 0.5], 'max_depth': [1, 2, 3]}
    boost = GradientBoostingClassifier(n_estimators=100, random_state=1)

    clf1 = GridSearchCV(boost, parameters)
    clf1 = clf1.fit(df1_x, df1_y)
    print("Digits Best Params boost:")
    print(clf1.best_params_)

    clf2 = GridSearchCV(boost, parameters)
    clf2 = clf2.fit(df2_x, df2_y)
    print("LoL Best Params boost:")
    print(clf2.best_params_)

    train_sizes, train_scores, valid_scores1 = learning_curve(GradientBoostingClassifier(random_state=1, 
                                                                                        n_estimators=100, 
                                                                                        learning_rate=0.3,
                                                                                        max_depth=2),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(GradientBoostingClassifier(random_state=1, 
                                                                                        n_estimators=100, 
                                                                                        learning_rate=0.3,
                                                                                        max_depth=2),
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
    plt.title("Boosting Learning Curve - Validation Scores (Digits Hyperparameters)")
    plt.xlabel("Training Examples [Percentage of Dataset]")
    plt.ylabel("Classification Score")
    plt.legend(["Digits-dataset", "LoL-dataset"])
    plt.savefig("Boost-1.png")

    train_sizes, train_scores, valid_scores1 = learning_curve(GradientBoostingClassifier(random_state=1, 
                                                                                        n_estimators=100, 
                                                                                        learning_rate=0.1,
                                                                                        max_depth=1),
                                                                                        train_sizes=np.linspace(0.1, 1, 10),
                                                                                        random_state=1, 
                                                                                        X=df1_x, y=df1_y, cv=5)
    train_sizes, train_scores, valid_scores2 = learning_curve(GradientBoostingClassifier(random_state=1, 
                                                                                        n_estimators=100, 
                                                                                        learning_rate=0.1,
                                                                                        max_depth=1),
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
    plt.title("Boosting Learning Curve - Validation Scores (LoL Hyperparameters)")
    plt.xlabel("Training Examples [Percentage of Dataset]")
    plt.ylabel("Classification Score")
    plt.legend(["Digits-dataset", "LoL-dataset"])
    plt.savefig("Boost-2.png")

if __name__ == "__main__":
    main()