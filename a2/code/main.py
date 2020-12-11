# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# our code
import utils

from knn import KNN

from naive_bayes import NaiveBayes

from decision_stump import DecisionStumpErrorRate, DecisionStumpEquality, DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest

from kmeans import Kmeans
from sklearn.cluster import DBSCAN


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question

    if question == "1":
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        model = DecisionTreeClassifier(
            max_depth=2, criterion='entropy', random_state=1)
        model.fit(X, y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)
        print("Training error: %.3f" % tr_error)
        print("Testing error: %.3f" % te_error)

    elif question == "1.1":
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]

        # TODO: mycode is from here
        depths = range(1, 16)
        tr_errors = []
        te_errors = []

        for depth in depths:
            model = DecisionTreeClassifier(
                max_depth=depth, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            te_errors.append(te_error)
            tr_errors.append(tr_error)

        plt.plot(depths, te_errors, label='Test Error')
        plt.plot(depths, tr_errors, label='Training Error')
        plt.xlabel('Depth of tree')
        plt.ylabel('Training error')
        plt.legend()
        fname = os.path.join("..", "figs", "q1.1.pdf")
        plt.savefig(fname)

    elif question == '1.2':
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X, y = dataset["X"], dataset["y"]
        n, d = X.shape

    elif question == '2.2':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]
        groupnames = dataset["groupnames"]
        wordlist = dataset["wordlist"]
        # 1
        print(wordlist[50])
        print()
        # 2
        example = X[500]
        for i in range(len(example)):
            if example[i]:
                print(wordlist[i])
        print()
        # 3
        print(groupnames[y[500]])

    elif question == '2.3':
        dataset = load_dataset("newsgroups.pkl")

        X = dataset["X"]
        y = dataset["y"]
        X_valid = dataset["Xvalidate"]
        y_valid = dataset["yvalidate"]

        print("d = %d" % X.shape[1])
        print("n = %d" % X.shape[0])
        print("t = %d" % X_valid.shape[0])
        print("Num classes = %d" % len(np.unique(y)))

        model = NaiveBayes(num_classes=4)
        model.fit(X, y)
        y_pred = model.predict(X_valid)
        v_error = np.mean(y_pred != y_valid)

        from sklearn.naive_bayes import BernoulliNB
        clf = BernoulliNB()
        clf.fit(X, y)
        sk_pred = clf.predict(X_valid)
        sk_error = np.mean(sk_pred != y_valid)

        print("Naive Bayes (ours) validation error: %.3f" % v_error)
        print("Naive Bayes (sklearn) validation error: %.3f" % sk_error)

    elif question == '3':
        with open(os.path.join('..', 'data', 'citiesSmall.pkl'), 'rb') as f:
            dataset = pickle.load(f)

        X = dataset['X']
        y = dataset['y']
        Xtest = dataset['Xtest']
        ytest = dataset['ytest']

        N = X.shape[0]
        T = Xtest.shape[0]

        model = KNN(1)
        model.fit(X, y)
        tr_pred = model.predict(X)
        te_pred = model.predict(Xtest)
        tr_error = np.sum(tr_pred != y)/N
        te_error = np.sum(te_pred != ytest)/T
        print("k=1")
        print("training error: ", tr_error)
        print("test error: ", te_error)
        print()

        # model = KNN(3)
        # model.fit(X, y)
        # tr_pred = model.predict(X)
        # te_pred = model.predict(Xtest)
        # tr_error = np.sum(tr_pred != y)/N
        # te_error = np.sum(te_pred != ytest)/T
        # print("k=3")
        # print("training error: ", tr_error)
        # print("test error: ", te_error)
        # print()

        # model = KNN(10)
        # model.fit(X, y)
        # tr_pred = model.predict(X)
        # te_pred = model.predict(Xtest)
        # tr_error = np.sum(tr_pred != y)/N
        # te_error = np.sum(te_pred != ytest)/T
        # print("k=10")
        # print("training error: ", tr_error)
        # print("test error: ", te_error)
        # print()

        knn = KNeighborsClassifier(1)
        knn.fit(X, y)

        # utils.plotClassifier(model, X, y)
        utils.plotClassifier(knn, X, y)
        # utils.plotClassifier(knn, Xtest, ytest)

    elif question == '4':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X, y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("Our random forest")
        evaluate_model(RandomForest(max_depth=100,
                                    num_trees=50))
        print("Sklearn random forest")
        evaluate_model(RandomForestClassifier(
            max_depth=100, n_estimators=50))

    elif question == '5':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)

        y = model.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")

        fname = os.path.join("..", "figs", "kmeans_basic.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.1':
        X = load_dataset('clusterData.pkl')['X']
        N, D = X.shape
        minError = np.Inf
        y = np.zeros(N)
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            error = model.error(X)
            if error < minError:
                minError = error
                y = model.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet")
        fname = os.path.join("..", "figs", "a2-q5.1-3.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    elif question == '5.2':
        X = load_dataset('clusterData.pkl')['X']
        errors = np.zeros(10)
        for k in range(10):
            minError = np.Inf
            for _ in range(50):
                model = Kmeans(k=k+1)
                model.fit(X)
                error = model.error(X)
                if error < minError:
                    minError = error
            errors[k] = minError

        plt.plot(range(1, 11), errors, label=f'Error')
        plt.xlabel('K')
        plt.ylabel('Error')
        plt.legend()
        fname = os.path.join("..", "figs", "q5.2-3.pdf")
        plt.savefig(fname)

    elif question == '5.3':
        X = load_dataset('clusterData2.pkl')['X']

        model = DBSCAN(eps=100, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet", s=5)
        fname = os.path.join("..", "figs", "density.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

        plt.xlim(-25, 25)
        plt.ylim(-15, 30)
        fname = os.path.join("..", "figs", "density2.png")
        plt.savefig(fname)
        print("Figure saved as '%s'" % fname)

    else:
        print("Unknown question: %s" % question)
